"""Carregamento de datasets para treinamento e avaliação.

Implementa a classe :class:`DeepfakeDataset` para o dataset
'140k Real and Fake Faces' do Kaggle, com split automático
treino/validação e transforms do torchvision.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

DEFAULT_ROOT: Path = Path("data/raw/real_vs_fake/real-vs-fake/train")
IMG_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Retorna transformações de treino com data augmentation.

    Inclui Resize, HorizontalFlip, ColorJitter (brightness/contrast)
    e normalização com estatísticas do ImageNet.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Retorna transformações determinísticas para validação/teste."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class DeepfakeDataset(Dataset):
    """Dataset de imagens reais (label=0) e falsas (label=1)."""

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        transform: Optional[transforms.Compose] = None,
        verbose: bool = True,
    ) -> None:
        """Inicializa o dataset varrendo as pastas ``real/`` e ``fake/``.

        Args:
            root: Pasta raiz contendo subpastas ``real`` e ``fake``.
            transform: Transformações do torchvision a aplicar nas imagens.
            verbose: Se True, imprime o total por classe.
        """
        self.root: Path = Path(root)
        self.transform: Optional[transforms.Compose] = transform
        self.samples: list[tuple[Path, int]] = []

        class_map: dict[str, int] = {"real": 0, "fake": 1}
        counts: dict[str, int] = {"real": 0, "fake": 0}

        for cls_name, label in class_map.items():
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                raise FileNotFoundError(f"Pasta não encontrada: {cls_dir}")
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in IMG_EXTENSIONS:
                    self.samples.append((img_path, label))
                    counts[cls_name] += 1

        if verbose:
            print(f"[DeepfakeDataset] real: {counts['real']} | fake: {counts['fake']} "
                  f"| total: {len(self.samples)}")

    def __len__(self) -> int:
        """Retorna o número total de amostras."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Retorna a tupla ``(image_tensor, label)`` para o índice dado."""
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Divide um dataset em treino/validação de forma reprodutível.

    Args:
        dataset: Dataset completo.
        val_ratio: Proporção destinada à validação (0-1).
        seed: Semente para o gerador aleatório.

    Returns:
        Tupla ``(train_subset, val_subset)``.
    """
    total = len(dataset)  # type: ignore[arg-type]
    val_size = int(total * val_ratio)
    train_size = total - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    return train_subset, val_subset


def build_datasets(
    root: str | Path = DEFAULT_ROOT,
    val_ratio: float = 0.2,
    image_size: int = 224,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Cria subsets de treino e validação com split 80/20.

    O subset de treino usa augmentation; o de validação usa
    transforms determinísticas.
    """
    train_full = DeepfakeDataset(root, transform=get_train_transforms(image_size))
    val_full = DeepfakeDataset(root, transform=get_eval_transforms(image_size), verbose=False)

    train_subset, _ = train_val_split(train_full, val_ratio=val_ratio, seed=seed)
    _, val_subset = train_val_split(val_full, val_ratio=val_ratio, seed=seed)
    return train_subset, val_subset


if __name__ == "__main__":
    # Script de teste: valida a estrutura e tipos retornados pelo dataset.
    print("=== Teste do DeepfakeDataset ===")
    train_ds, val_ds = build_datasets()
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    image, label = train_ds[0]
    assert isinstance(image, torch.Tensor), "image deveria ser torch.Tensor"
    assert image.shape == (3, 224, 224), f"shape inesperado: {image.shape}"
    assert label in (0, 1), f"label inválido: {label}"
    print(f"Amostra OK -> shape={tuple(image.shape)}, dtype={image.dtype}, label={label}")
    print("Teste concluido com sucesso.")
