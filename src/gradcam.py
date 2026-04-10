"""Grad-CAM para o modelo DeepShield.

Gera mapas de ativação (heatmaps) que mostram *onde* o modelo focou para
classificar uma imagem como real ou fake. Baseado no paper:

    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.

Funções principais:

* :class:`GradCAM` — extrai ativações + gradientes da última camada conv.
* :func:`generate_heatmap` — produz o heatmap sobreposto à imagem.
* :func:`process_batch` — processa múltiplas imagens de uma vez.

Exemplo::

    python -m src.gradcam --image data/raw/real_vs_fake/real-vs-fake/train/fake/00001.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from PIL import Image

from .model import DeepShieldModel
from .preprocessing import get_eval_transforms

LABELS: dict[int, str] = {0: "real", 1: "fake"}


# --------------------------------------------------------------------- #
# Grad-CAM core
# --------------------------------------------------------------------- #
class GradCAM:
    """Extrator de Grad-CAM para :class:`DeepShieldModel`.

    Registra hooks na última camada convolucional do backbone
    (``forward_features``) para capturar ativações e gradientes.
    """

    def __init__(self, model: DeepShieldModel, device: torch.device | None = None) -> None:
        """Inicializa o Grad-CAM.

        Args:
            model: Modelo DeepShield já carregado.
            device: Dispositivo de execução.
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Registra hooks na última camada conv do EfficientNet (bn2)
        target_layer = self.model.backbone.bn2
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(
        self,
        module: torch.nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """Hook de forward: salva ativações."""
        self._activations = output.detach()

    def _save_gradient(
        self,
        module: torch.nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """Hook de backward: salva gradientes."""
        self._gradients = grad_output[0].detach()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[npt.NDArray[np.floating], int, float]:
        """Gera o mapa Grad-CAM para um input.

        Args:
            input_tensor: Tensor (1, 3, H, W) normalizado.
            target_class: Classe alvo. Se None, usa a classe predita.

        Returns:
            Tupla ``(cam, predicted_class, confidence)``.
            ``cam`` é um array (H, W) normalizado em [0, 1].
        """
        input_tensor = input_tensor.to(self.device)
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, target_class].item())

        # Backward na classe alvo
        self.model.zero_grad()
        output[0, target_class].backward()

        # Grad-CAM: média dos gradientes por canal → pesos → soma ponderada
        gradients = self._gradients  # (1, C, h, w)
        activations = self._activations  # (1, C, h, w)
        assert gradients is not None and activations is not None

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Resize para tamanho da entrada
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normaliza para [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam, target_class, confidence


# --------------------------------------------------------------------- #
# Overlay / Heatmap
# --------------------------------------------------------------------- #
def overlay_heatmap(
    image: npt.NDArray[np.uint8],
    cam: npt.NDArray[np.floating],
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> npt.NDArray[np.uint8]:
    """Sobrepõe o heatmap Grad-CAM na imagem original.

    Args:
        image: Imagem BGR (H, W, 3) em uint8.
        cam: Mapa de ativação (H, W) normalizado em [0, 1].
        alpha: Transparência do heatmap (0 = só imagem, 1 = só heatmap).
        colormap: Colormap do OpenCV.

    Returns:
        Imagem BGR com heatmap sobreposto.
    """
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    blended: npt.NDArray[np.uint8] = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return blended


# --------------------------------------------------------------------- #
# Funções de alto nível
# --------------------------------------------------------------------- #
def generate_heatmap(
    image_path: str | Path,
    model: DeepShieldModel,
    device: torch.device | None = None,
    alpha: float = 0.5,
) -> tuple[npt.NDArray[np.uint8], str, float, npt.NDArray[np.floating]]:
    """Gera heatmap Grad-CAM para uma única imagem.

    Args:
        image_path: Caminho da imagem.
        model: Modelo treinado.
        device: Dispositivo de inferência.
        alpha: Transparência do heatmap.

    Returns:
        Tupla ``(overlay, label, confidence, raw_cam)``.
    """
    device = device or next(model.parameters()).device
    gradcam = GradCAM(model, device)

    # Prepara input
    transform = get_eval_transforms()
    pil_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0)

    # Gera CAM
    cam, pred_class, confidence = gradcam(input_tensor)

    # Overlay na imagem original
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    overlay = overlay_heatmap(img_bgr, cam, alpha=alpha)

    return overlay, LABELS[pred_class], confidence, cam


def process_batch(
    image_paths: list[str | Path],
    model: DeepShieldModel,
    output_dir: str | Path,
    device: torch.device | None = None,
    alpha: float = 0.5,
) -> list[dict[str, str | float]]:
    """Processa múltiplas imagens e salva resultados como PNG.

    Para cada imagem gera um PNG com 3 painéis: original | heatmap | overlay.
    Salva em ``output_dir/gradcam_<nome>.png``.

    Args:
        image_paths: Lista de caminhos de imagens.
        model: Modelo treinado.
        output_dir: Diretório de saída.
        device: Dispositivo de inferência.
        alpha: Transparência do heatmap.

    Returns:
        Lista de dicionários com metadados de cada resultado.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = device or next(model.parameters()).device
    gradcam = GradCAM(model, device)
    transform = get_eval_transforms()

    results: list[dict[str, str | float]] = []

    for img_path in image_paths:
        img_path = Path(img_path)

        # Inferência + Grad-CAM
        pil_img = Image.open(img_path).convert("RGB")
        input_tensor = transform(pil_img).unsqueeze(0)
        cam, pred_class, confidence = gradcam(input_tensor)

        # Imagem original em BGR
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        # Heatmap colorido (sem overlay)
        cam_resized = cv2.resize(cam.astype(np.float32), (w, h))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

        # Overlay
        overlay = overlay_heatmap(img_bgr, cam, alpha=alpha)

        # Monta painel: original | heatmap | overlay
        label = LABELS[pred_class]
        panel = np.hstack([img_bgr, heatmap_colored, overlay])

        # Adiciona texto
        text = f"{label.upper()} ({confidence:.1%})"
        cv2.putText(
            panel, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Salva
        out_name = f"gradcam_{img_path.stem}.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), panel, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        results.append({
            "image": str(img_path),
            "prediction": label,
            "confidence": confidence,
            "output": str(out_path),
        })
        print(f"  {img_path.name} -> {label} ({confidence:.1%}) -> {out_name}")

    return results


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM do DeepShield.")
    parser.add_argument("--image", type=str, help="Caminho de uma imagem.")
    parser.add_argument("--image_dir", type=str, help="Diretório com imagens.")
    parser.add_argument("--weights", type=str, default="models/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="outputs/gradcam")
    parser.add_argument("--max_images", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada CLI."""
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carrega modelo
    model = DeepShieldModel(pretrained=False).to(device)
    weights_path = Path(args.weights)
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Pesos carregados: {weights_path}")
    else:
        print(f"AVISO: pesos não encontrados em {weights_path}, usando modelo sem treino.")

    # Coleta imagens
    paths: list[Path] = []
    if args.image:
        paths.append(Path(args.image))
    elif args.image_dir:
        img_dir = Path(args.image_dir)
        paths = sorted(img_dir.glob("*"))[:args.max_images]
    else:
        # Fallback: pega amostras de real e fake
        data = Path("data/raw/real_vs_fake/real-vs-fake/train")
        paths = sorted((data / "real").glob("*"))[:5] + sorted((data / "fake").glob("*"))[:5]

    print(f"Processando {len(paths)} imagens...")
    results = process_batch(paths, model, args.output_dir, device, args.alpha)
    print(f"\n{len(results)} resultados salvos em {args.output_dir}/")


if __name__ == "__main__":
    main()
