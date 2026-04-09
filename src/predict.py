"""Inferência do modelo DeepShield.

Carrega pesos treinados e prediz se uma imagem é real ou falsa.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .model import DeepShieldModel
from .preprocessing import get_eval_transforms


def predict_image(
    image_path: str | Path,
    weights_path: str | Path,
    device: torch.device | None = None,
) -> tuple[str, float]:
    """Prediz a classe de uma imagem.

    Args:
        image_path: Caminho da imagem.
        weights_path: Caminho do arquivo .pth com os pesos.
        device: Dispositivo de inferência.

    Returns:
        Tupla (rótulo, confiança).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepShieldModel(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    transform = get_eval_transforms()
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(image), dim=1)[0]
    idx = int(torch.argmax(probs).item())
    label = "real" if idx == 0 else "fake"
    return label, float(probs[idx].item())
