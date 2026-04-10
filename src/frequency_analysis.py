"""Análise de frequência para detecção de deepfakes.

Deepfakes introduzem artefatos no domínio de frequência (padrões de
alta frequência atenuados, grid artifacts de upsampling, etc.). Este
módulo usa a FFT 2D para extrair e visualizar essas diferenças.

Funções principais:

* :func:`compute_spectrum` — converte imagem para espectro de potência.
* :func:`extract_frequency_features` — vetor de features do espectro.
* :func:`compute_mean_spectrum` — espectro médio de uma lista de imagens.
* :func:`plot_image_vs_spectrum` — visualização lado a lado.
* :func:`plot_mean_spectrum_comparison` — comparação real vs fake.

Exemplo::

    python -m src.frequency_analysis
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


# --------------------------------------------------------------------- #
# Tipo reutilizado
# --------------------------------------------------------------------- #
NDArrayFloat = npt.NDArray[np.floating]


# --------------------------------------------------------------------- #
# FFT / Espectro
# --------------------------------------------------------------------- #
def compute_spectrum(image: npt.NDArray[np.uint8]) -> NDArrayFloat:
    """Converte uma imagem BGR/RGB para seu espectro de potência via FFT 2D.

    Passos:
        1. Converte para escala de cinza.
        2. Aplica FFT 2D e centraliza (shift).
        3. Calcula magnitude em escala log.

    Args:
        image: Imagem (H, W, 3) ou (H, W) em uint8.

    Returns:
        Espectro de potência normalizado em [0, 1], shape (H, W).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray_float = np.float32(gray)
    fft = np.fft.fft2(gray_float)
    fft_shift = np.fft.fftshift(fft)
    magnitude: NDArrayFloat = np.log1p(np.abs(fft_shift))

    # Normaliza para [0, 1]
    mag_min, mag_max = magnitude.min(), magnitude.max()
    if mag_max - mag_min > 0:
        magnitude = (magnitude - mag_min) / (mag_max - mag_min)
    return magnitude


def compute_azimuthal_profile(spectrum: NDArrayFloat) -> NDArrayFloat:
    """Calcula o perfil radial médio (azimuthal average) do espectro.

    Agrupa pixels por distância ao centro e tira a média — útil para
    comparar distribuição de energia por frequência espacial.

    Args:
        spectrum: Espectro 2D (H, W), tipicamente saída de :func:`compute_spectrum`.

    Returns:
        Array 1D com a média radial, comprimento = metade da menor dimensão.
    """
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_r = min(cy, cx)
    profile = np.array(
        [spectrum[radius == r].mean() for r in range(max_r)],
        dtype=np.float64,
    )
    return profile


# --------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------- #
def extract_frequency_features(
    image: npt.NDArray[np.uint8],
    n_bins: int = 64,
) -> NDArrayFloat:
    """Extrai um vetor de features do espectro de frequência.

    Divide o perfil radial em ``n_bins`` faixas equidistantes e calcula
    a energia média em cada uma. Pode ser concatenado às features da CNN
    como input complementar.

    Args:
        image: Imagem BGR/RGB (H, W, 3) ou grayscale (H, W).
        n_bins: Número de bins no perfil radial.

    Returns:
        Array 1D de shape ``(n_bins,)`` com as features de frequência.
    """
    spectrum = compute_spectrum(image)
    profile = compute_azimuthal_profile(spectrum)

    # Redimensiona o perfil para n_bins via média por bin
    bin_edges = np.linspace(0, len(profile), n_bins + 1, dtype=int)
    features = np.array(
        [profile[bin_edges[i] : bin_edges[i + 1]].mean() for i in range(n_bins)],
        dtype=np.float64,
    )
    return features


# --------------------------------------------------------------------- #
# Espectro médio de um conjunto
# --------------------------------------------------------------------- #
def compute_mean_spectrum(
    image_dir: str | Path,
    max_images: int = 500,
    target_size: tuple[int, int] = (224, 224),
) -> NDArrayFloat:
    """Calcula o espectro de potência médio de imagens em um diretório.

    Args:
        image_dir: Diretório contendo imagens (jpg/png).
        max_images: Número máximo de imagens a processar.
        target_size: Tamanho (H, W) para resize antes da FFT.

    Returns:
        Espectro médio de shape ``target_size``.
    """
    image_dir = Path(image_dir)
    paths = sorted(image_dir.glob("*"))[:max_images]

    spectra: list[NDArrayFloat] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (target_size[1], target_size[0]))
        spectra.append(compute_spectrum(img))

    if not spectra:
        raise ValueError(f"Nenhuma imagem encontrada em {image_dir}")

    return np.mean(spectra, axis=0)


# --------------------------------------------------------------------- #
# Visualizações
# --------------------------------------------------------------------- #
def plot_image_vs_spectrum(
    image_path: str | Path,
    save_path: str | Path | None = None,
) -> None:
    """Plota imagem original ao lado do seu espectro de frequência.

    Args:
        image_path: Caminho da imagem.
        save_path: Se fornecido, salva o plot neste caminho.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    spectrum = compute_spectrum(img)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Imagem Original")
    axes[0].axis("off")

    axes[1].imshow(spectrum, cmap="inferno")
    axes[1].set_title("Espectro de Frequência (FFT 2D)")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.show()


def plot_mean_spectrum_comparison(
    real_dir: str | Path,
    fake_dir: str | Path,
    max_images: int = 500,
    save_path: str | Path | None = None,
) -> None:
    """Plota comparação: espectro médio real vs fake + diferença + perfil radial.

    Gera 4 subplots:
        1. Espectro médio — REAL
        2. Espectro médio — FAKE
        3. Diferença absoluta (|real − fake|)
        4. Perfil radial (azimuthal average) sobrepostos

    Args:
        real_dir: Diretório de imagens reais.
        fake_dir: Diretório de imagens fake.
        max_images: Máximo de imagens por classe.
        save_path: Se fornecido, salva o plot.
    """
    print(f"Calculando espectro médio de REAL ({max_images} imgs)...")
    real_spec = compute_mean_spectrum(real_dir, max_images=max_images)
    print(f"Calculando espectro médio de FAKE ({max_images} imgs)...")
    fake_spec = compute_mean_spectrum(fake_dir, max_images=max_images)

    diff = np.abs(real_spec - fake_spec)
    real_profile = compute_azimuthal_profile(real_spec)
    fake_profile = compute_azimuthal_profile(fake_spec)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    axes[0].imshow(real_spec, cmap="inferno")
    axes[0].set_title("Espectro Médio — REAL")
    axes[0].axis("off")

    axes[1].imshow(fake_spec, cmap="inferno")
    axes[1].set_title("Espectro Médio — FAKE")
    axes[1].axis("off")

    axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("|REAL − FAKE|")
    axes[2].axis("off")

    axes[3].plot(real_profile, label="Real", linewidth=1.5)
    axes[3].plot(fake_profile, label="Fake", linewidth=1.5)
    axes[3].set_title("Perfil Radial")
    axes[3].set_xlabel("Frequência espacial")
    axes[3].set_ylabel("Energia média")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("DeepShield — Análise de Frequência: Real vs Fake", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.show()


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    DATA_ROOT = Path("data/raw/real_vs_fake/real-vs-fake/train")
    real_dir = DATA_ROOT / "real"
    fake_dir = DATA_ROOT / "fake"

    # 1) Exemplo: imagem vs espectro
    sample_img = next(real_dir.glob("*"))
    print(f"Amostra: {sample_img.name}")
    plot_image_vs_spectrum(sample_img)

    # 2) Feature extraction de uma imagem
    img = cv2.imread(str(sample_img))
    feats = extract_frequency_features(img, n_bins=64)
    print(f"Frequency features shape: {feats.shape}")
    print(f"Primeiros 8 valores: {feats[:8].round(4)}")

    # 3) Comparação real vs fake (espectro médio, 200 imgs por classe)
    plot_mean_spectrum_comparison(real_dir, fake_dir, max_images=200)
