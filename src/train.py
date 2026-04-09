"""Pipeline completo de treinamento do DeepShield.

Executa transfer learning em duas fases sobre o EfficientNet-B0:

1. **Fase 1** – backbone congelado, treina apenas o classificador.
2. **Fase 2** – descongela os últimos blocos e faz fine-tuning com LR menor.

Inclui AdamW + CosineAnnealingLR, métricas por época (loss, accuracy,
precision, recall, F1), early stopping, salvamento do melhor modelo e
histórico em JSON.

Exemplo::

    python -m src.train --epochs 15 --batch_size 64 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .dataset import DeepfakeDataset, build_datasets
from .model import DeepShieldModel


# --------------------------------------------------------------------- #
# Estruturas auxiliares
# --------------------------------------------------------------------- #
@dataclass
class EpochMetrics:
    """Métricas agregadas de uma época."""

    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass
class TrainHistory:
    """Histórico de treinamento acumulado por época."""

    train: list[dict[str, float]] = field(default_factory=list)
    val: list[dict[str, float]] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    phase: list[int] = field(default_factory=list)

    def log(
        self,
        train_m: EpochMetrics,
        val_m: EpochMetrics,
        lr: float,
        phase: int,
    ) -> None:
        self.train.append(train_m.as_dict())
        self.val.append(val_m.as_dict())
        self.lr.append(lr)
        self.phase.append(phase)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {"train": self.train, "val": self.val, "lr": self.lr, "phase": self.phase},
                f,
                indent=2,
            )


# --------------------------------------------------------------------- #
# Utilitários
# --------------------------------------------------------------------- #
def auto_device(requested: str) -> torch.device:
    """Retorna o device apropriado (cuda se disponível)."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def compute_class_weights(subset: Subset) -> torch.Tensor:
    """Calcula pesos inversamente proporcionais à frequência das classes."""
    base: DeepfakeDataset = subset.dataset  # type: ignore[assignment]
    labels = [base.samples[i][1] for i in subset.indices]  # type: ignore[attr-defined]
    counts = Counter(labels)
    total = sum(counts.values())
    num_classes = len(counts)
    weights = [total / (num_classes * counts[c]) for c in sorted(counts)]
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(
    loss: float, y_true: list[int], y_pred: list[int]
) -> EpochMetrics:
    """Calcula métricas agregadas (macro) a partir das listas de predição."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    accuracy = sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(len(y_true), 1)
    return EpochMetrics(
        loss=loss,
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


# --------------------------------------------------------------------- #
# Loops de treino/validação
# --------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    desc: str,
) -> EpochMetrics:
    """Executa uma época de treinamento."""
    model.train()
    running_loss = 0.0
    n_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
        preds = outputs.argmax(dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())
        pbar.set_postfix(loss=f"{running_loss / n_samples:.4f}")

    return compute_metrics(running_loss / n_samples, y_true, y_pred)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "val",
) -> EpochMetrics:
    """Avalia o modelo no loader fornecido."""
    model.eval()
    running_loss = 0.0
    n_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        n_samples += images.size(0)
        preds = outputs.argmax(dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

    return compute_metrics(running_loss / n_samples, y_true, y_pred)


# --------------------------------------------------------------------- #
# Loop principal
# --------------------------------------------------------------------- #
def run_phase(
    model: DeepShieldModel,
    phase_id: int,
    epochs: int,
    lr: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    save_dir: Path,
    history: TrainHistory,
    patience: int,
    best_state: dict[str, Any],
) -> None:
    """Executa uma fase de treinamento (com early stopping próprio)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        header = f"[Fase {phase_id}] Epoch {epoch}/{epochs}"
        print(f"\n{header}")

        train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, device, desc=f"{header} train"
        )
        val_m = evaluate(model, val_loader, criterion, device, desc=f"{header} val")
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history.log(train_m, val_m, current_lr, phase_id)

        print(
            f"  train -> loss={train_m.loss:.4f} acc={train_m.accuracy:.4f} "
            f"f1={train_m.f1:.4f}"
        )
        print(
            f"  val   -> loss={val_m.loss:.4f} acc={val_m.accuracy:.4f} "
            f"prec={val_m.precision:.4f} rec={val_m.recall:.4f} f1={val_m.f1:.4f}"
        )

        if val_m.f1 > best_state["best_f1"]:
            best_state["best_f1"] = val_m.f1
            best_state["best_epoch"] = len(history.val)
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  ✔ novo melhor F1={val_m.f1:.4f} salvo em best_model.pth")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            print(f"  sem melhora ({epochs_without_improve}/{patience})")
            if epochs_without_improve >= patience:
                print(f"  early stopping acionado na fase {phase_id}.")
                return


def main(args: argparse.Namespace) -> None:
    """Função principal do pipeline de treinamento."""
    device = auto_device(args.device)
    print(f"Device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Datasets / Loaders
    train_subset, val_subset = build_datasets(
        val_ratio=0.2, image_size=224, seed=42
    )
    print(f"Train: {len(train_subset)} | Val: {len(val_subset)}")

    num_workers = 0 if device.type == "cpu" else min(4, (args.num_workers or 4))
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    # Modelo + Loss com class weights
    model = DeepShieldModel(pretrained=True).to(device)
    class_weights = compute_class_weights(train_subset).to(device)
    print(f"Class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = TrainHistory()
    best_state: dict[str, Any] = {"best_f1": -1.0, "best_epoch": 0}

    # Fase 1: backbone congelado
    model.freeze_backbone()
    model.summary()
    run_phase(
        model,
        phase_id=1,
        epochs=args.phase1_epochs,
        lr=args.lr,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        save_dir=save_dir,
        history=history,
        patience=args.patience,
        best_state=best_state,
    )

    # Fase 2: fine-tuning dos últimos blocos
    model.unfreeze_last_blocks(n=3)
    model.summary()
    run_phase(
        model,
        phase_id=2,
        epochs=args.epochs - args.phase1_epochs,
        lr=args.lr * 0.1,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        save_dir=save_dir,
        history=history,
        patience=args.patience,
        best_state=best_state,
    )

    history.save(save_dir / "history.json")
    print(
        f"\nTreinamento concluido. Melhor F1={best_state['best_f1']:.4f} "
        f"(epoch global {best_state['best_epoch']}). "
        f"Artefatos em {save_dir}."
    )


def parse_args() -> argparse.Namespace:
    """Parseia os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Treina o DeepShield.")
    parser.add_argument("--epochs", type=int, default=15, help="Total de épocas (fase1+fase2).")
    parser.add_argument("--phase1_epochs", type=int, default=5, help="Épocas da fase 1.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping (épocas).")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
