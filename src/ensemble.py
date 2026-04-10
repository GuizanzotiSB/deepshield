"""Ensemble do DeepShield — combina CNN, análise de frequência e consistência.

Funde três sinais complementares para produzir uma pontuação final
de probabilidade de deepfake:

1. **CNN Score** — probabilidade ``fake`` do EfficientNet-B0.
2. **FFT Score** — anomalia no espectro de frequência (perfil radial).
3. **Consistência** — artefatos de borda e textura via OpenCV.

Os pesos padrão são 0.60 / 0.25 / 0.15, ajustáveis por parâmetro.

Exemplo::

    python -m src.ensemble --image foto.jpg --weights models/best_model.pth
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from PIL import Image

from .frequency_analysis import compute_spectrum, compute_azimuthal_profile
from .gradcam import GradCAM, overlay_heatmap
from .model import DeepShieldModel
from .preprocessing import get_eval_transforms


# --------------------------------------------------------------------- #
# Resultado estruturado
# --------------------------------------------------------------------- #
@dataclass
class AnalysisResult:
    """Resultado completo da análise ensemble."""

    deepfake_score: float          # 0-100 %
    confidence_level: str          # Baixo / Médio / Alto
    cnn_score: float               # 0-1
    fft_score: float               # 0-1
    consistency_score: float       # 0-1
    prediction: str                # "real" ou "fake"
    gradcam: npt.NDArray[np.floating]   # mapa CAM (H, W) em [0,1]
    overlay: npt.NDArray[np.uint8]      # imagem BGR com heatmap

    def summary(self) -> str:
        """Retorna resumo textual da análise."""
        lines = [
            f"Resultado     : {self.prediction.upper()}",
            f"Score final   : {self.deepfake_score:.1f}%",
            f"Confiança     : {self.confidence_level}",
            f"  CNN         : {self.cnn_score:.1%}",
            f"  Frequência  : {self.fft_score:.1%}",
            f"  Consistência: {self.consistency_score:.1%}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------- #
# Sub-analisadores
# --------------------------------------------------------------------- #
def _cnn_score(
    model: DeepShieldModel,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> float:
    """Retorna a probabilidade da classe ``fake`` via softmax."""
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor.to(device))
        probs = F.softmax(logits, dim=1)
    return float(probs[0, 1].item())  # indice 1 = fake


def _fft_score(image_bgr: npt.NDArray[np.uint8]) -> float:
    """Estima anomalia de frequência comparando energia alta vs baixa.

    Deepfakes tendem a ter menos energia em altas frequências (suavização
    por upsampling) ou picos anômalos (grid artifacts). Medimos a razão
    entre energia na metade superior e inferior do perfil radial.
    Um valor atípico (muito baixo) indica manipulação.

    Returns:
        Score em [0, 1]: 0 = perfil normal, 1 = muito anômalo.
    """
    spectrum = compute_spectrum(image_bgr)
    profile = compute_azimuthal_profile(spectrum)

    if len(profile) < 4:
        return 0.0

    mid = len(profile) // 2
    low_energy = profile[:mid].mean()
    high_energy = profile[mid:].mean()

    if low_energy == 0:
        return 0.0

    # Razão alta/baixa — imagens reais ~0.3-0.5, fakes ~0.1-0.25
    ratio = high_energy / low_energy

    # Mapeia para [0, 1]: quanto menor a razão, mais suspeito
    # Sigmoid invertida centrada em 0.35 (limiar empírico)
    score = 1.0 / (1.0 + np.exp(15 * (ratio - 0.35)))
    return float(np.clip(score, 0.0, 1.0))


def _consistency_score(image_bgr: npt.NDArray[np.uint8]) -> float:
    """Detecta artefatos de consistência via análise de bordas e textura.

    Combina três sinais:
    1. Variância do Laplaciano — foco inconsistente.
    2. Descontinuidades no mapa de bordas (Canny) — costuras de blending.
    3. Assimetria da Error Level Analysis (ELA) — regiões recomprimidas.

    Returns:
        Score em [0, 1]: 0 = consistente, 1 = muitos artefatos.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- 1. Variância local do Laplaciano (foco inconsistente) ---
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    block_size = max(h, w) // 8
    if block_size < 8:
        block_size = 8
    block_vars: list[float] = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = laplacian[y : y + block_size, x : x + block_size]
            block_vars.append(float(block.var()))

    if len(block_vars) < 2:
        focus_score = 0.0
    else:
        arr = np.array(block_vars)
        cv_focus = arr.std() / (arr.mean() + 1e-8)  # coeficiente de variação
        focus_score = float(np.clip(cv_focus / 3.0, 0.0, 1.0))

    # --- 2. Descontinuidades de borda ---
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean() / 255.0
    # Bordas demais ou de menos é suspeito — distância da faixa normal 0.04-0.12
    if edge_density < 0.04:
        edge_score = (0.04 - edge_density) / 0.04
    elif edge_density > 0.12:
        edge_score = min((edge_density - 0.12) / 0.12, 1.0)
    else:
        edge_score = 0.0

    # --- 3. Error Level Analysis simplificada ---
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, 90]
    _, encoded = cv2.imencode(".jpg", image_bgr, encode_param)
    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    ela = cv2.absdiff(image_bgr, recompressed).astype(np.float32)
    ela_gray = cv2.cvtColor(ela.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # Blocos com ELA muito acima da média indicam edição localizada
    ela_mean = ela_gray.mean()
    ela_std = ela_gray.std()
    if ela_mean < 1e-8:
        ela_score = 0.0
    else:
        ela_cv = ela_std / ela_mean
        ela_score = float(np.clip(ela_cv / 4.0, 0.0, 1.0))

    # Média ponderada dos 3 sinais
    combined = 0.4 * focus_score + 0.35 * edge_score + 0.25 * ela_score
    return float(np.clip(combined, 0.0, 1.0))


# --------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------- #
DEFAULT_WEIGHTS: tuple[float, float, float] = (0.60, 0.25, 0.15)


def classify_confidence(score: float) -> str:
    """Classifica o nível de confiança com base no score final."""
    if score < 30 or score > 85:
        return "Alto"
    if score < 40 or score > 70:
        return "Médio"
    return "Baixo"


def analyze_image(
    image_path: str | Path,
    model: DeepShieldModel,
    device: torch.device | None = None,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    gradcam_alpha: float = 0.5,
) -> AnalysisResult:
    """Executa a análise ensemble completa em uma imagem.

    Args:
        image_path: Caminho da imagem.
        model: Modelo DeepShield com pesos carregados.
        device: Dispositivo de inferência.
        weights: Tupla ``(w_cnn, w_fft, w_consistency)``. Deve somar 1.0.
        gradcam_alpha: Transparência do overlay Grad-CAM.

    Returns:
        :class:`AnalysisResult` com todos os sub-scores e visualizações.
    """
    image_path = Path(image_path)
    device = device or next(model.parameters()).device
    w_cnn, w_fft, w_cons = weights

    # Carrega imagem
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    transform = get_eval_transforms()
    pil_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0)

    # 1) CNN Score
    cnn = _cnn_score(model, input_tensor, device)

    # 2) FFT Score
    fft = _fft_score(img_bgr)

    # 3) Consistência Score
    cons = _consistency_score(img_bgr)

    # Score final (0-100)
    final = (w_cnn * cnn + w_fft * fft + w_cons * cons) * 100.0
    final = float(np.clip(final, 0.0, 100.0))

    # Confiança
    confidence_level = classify_confidence(final)

    # Predição
    prediction = "fake" if final >= 50.0 else "real"

    # Grad-CAM
    gradcam = GradCAM(model, device)
    cam, _, _ = gradcam(input_tensor)
    overlay = overlay_heatmap(img_bgr, cam, alpha=gradcam_alpha)

    return AnalysisResult(
        deepfake_score=final,
        confidence_level=confidence_level,
        cnn_score=cnn,
        fft_score=fft,
        consistency_score=cons,
        prediction=prediction,
        gradcam=cam,
        overlay=overlay,
    )


def analyze_batch(
    image_paths: list[str | Path],
    model: DeepShieldModel,
    output_dir: str | Path,
    device: torch.device | None = None,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> list[AnalysisResult]:
    """Processa múltiplas imagens e salva resultados.

    Para cada imagem salva o overlay Grad-CAM e imprime o resumo.

    Args:
        image_paths: Lista de caminhos.
        model: Modelo treinado.
        output_dir: Diretório de saída.
        device: Dispositivo.
        weights: Pesos do ensemble.

    Returns:
        Lista de :class:`AnalysisResult`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[AnalysisResult] = []
    for img_path in image_paths:
        img_path = Path(img_path)
        result = analyze_image(img_path, model, device, weights)
        results.append(result)

        # Salva overlay
        out_path = output_dir / f"ensemble_{img_path.stem}.png"
        cv2.imwrite(str(out_path), result.overlay, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        print(f"\n--- {img_path.name} ---")
        print(result.summary())

    return results


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepShield Ensemble Analysis.")
    parser.add_argument("--image", type=str, help="Caminho de uma imagem.")
    parser.add_argument("--image_dir", type=str, help="Diretório com imagens.")
    parser.add_argument("--weights_path", type=str, default="models/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="outputs/ensemble")
    parser.add_argument("--max_images", type=int, default=10)
    parser.add_argument(
        "--ensemble_weights", type=float, nargs=3, default=[0.60, 0.25, 0.15],
        metavar=("CNN", "FFT", "CONS"),
        help="Pesos: CNN FFT Consistência (devem somar 1.0)",
    )
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada CLI."""
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepShieldModel(pretrained=False).to(device)
    wp = Path(args.weights_path)
    if wp.exists():
        model.load_state_dict(torch.load(wp, map_location=device))
        print(f"Pesos carregados: {wp}")
    else:
        print(f"AVISO: {wp} não encontrado, usando modelo sem treino.")

    # Coleta imagens
    paths: list[Path] = []
    if args.image:
        paths = [Path(args.image)]
    elif args.image_dir:
        d = Path(args.image_dir)
        paths = sorted(d.glob("*"))[:args.max_images]
    else:
        data = Path("data/raw/real_vs_fake/real-vs-fake/train")
        paths = sorted((data / "real").glob("*"))[:3] + sorted((data / "fake").glob("*"))[:3]

    ensemble_w = tuple(args.ensemble_weights)
    print(f"Device: {device} | Pesos ensemble: CNN={ensemble_w[0]} FFT={ensemble_w[1]} CONS={ensemble_w[2]}")
    print(f"Processando {len(paths)} imagens...\n")

    analyze_batch(paths, model, args.output_dir, device, ensemble_w)  # type: ignore[arg-type]
    print(f"\nResultados salvos em {args.output_dir}/")


if __name__ == "__main__":
    main()
