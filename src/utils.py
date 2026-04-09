"""Funções auxiliares diversas do DeepShield."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Define seeds para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Retorna o dispositivo disponível (cuda ou cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    """Garante que um diretório exista, criando-o se necessário."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
