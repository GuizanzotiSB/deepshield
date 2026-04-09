"""Smoke test rápido do pipeline DeepShield.

Treina em CPU sobre um subconjunto muito pequeno (200 imagens, 1 época)
apenas para garantir que dataset, modelo, forward, backward, loss e
métricas funcionam de ponta a ponta sem erros.

Uso::

    python -m tests.smoke_test
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from src.dataset import DeepfakeDataset, get_train_transforms
from src.model import DeepShieldModel
from src.train import compute_metrics, evaluate, train_one_epoch


def build_tiny_subset(total: int = 200, seed: int = 0) -> Subset:
    """Cria um Subset pequeno e balanceado para o smoke test."""
    ds = DeepfakeDataset(transform=get_train_transforms(), verbose=False)
    reals = [i for i, (_, lbl) in enumerate(ds.samples) if lbl == 0]
    fakes = [i for i, (_, lbl) in enumerate(ds.samples) if lbl == 1]
    rng = random.Random(seed)
    half = total // 2
    indices = rng.sample(reals, half) + rng.sample(fakes, half)
    rng.shuffle(indices)
    return Subset(ds, indices)


def main() -> None:
    """Executa o smoke test."""
    print("=== DeepShield smoke test ===")
    device = torch.device("cpu")

    subset = build_tiny_subset(total=200)
    split = len(subset) - 40
    train_subset = Subset(subset.dataset, subset.indices[:split])
    val_subset = Subset(subset.dataset, subset.indices[split:])
    print(f"Train: {len(train_subset)} | Val: {len(val_subset)}")

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=0)

    model = DeepShieldModel(pretrained=False).to(device)
    model.freeze_backbone()

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=1e-4,
    )

    train_m = train_one_epoch(
        model, train_loader, criterion, optimizer, device, desc="smoke train"
    )
    val_m = evaluate(model, val_loader, criterion, device, desc="smoke val")

    print(
        f"train -> loss={train_m.loss:.4f} acc={train_m.accuracy:.4f} f1={train_m.f1:.4f}"
    )
    print(
        f"val   -> loss={val_m.loss:.4f} acc={val_m.accuracy:.4f} f1={val_m.f1:.4f}"
    )

    assert train_m.loss > 0, "train loss deveria ser > 0"
    assert 0.0 <= val_m.accuracy <= 1.0, "accuracy fora do intervalo"
    _ = compute_metrics(0.1, [0, 1], [0, 1])  # sanity check da função
    print("Smoke test OK.")


if __name__ == "__main__":
    main()
