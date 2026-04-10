"""Interface web do DeepShield.

Aplicação Streamlit com upload de imagem, análise ensemble (CNN + FFT +
Consistência), Grad-CAM e visualizações interativas.

Uso::

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

# Garante que imports de src/ funcionem
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ensemble import AnalysisResult, analyze_image  # noqa: E402
from src.frequency_analysis import compute_spectrum  # noqa: E402
from src.model import DeepShieldModel  # noqa: E402

WEIGHTS_PATH = ROOT / "models" / "best_model.pth"


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
@st.cache_resource
def load_model() -> tuple[DeepShieldModel, torch.device]:
    """Carrega o modelo uma única vez e mantém em cache."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepShieldModel(pretrained=False).to(device)
    if WEIGHTS_PATH.exists():
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    return model, device


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fig_to_image(fig: plt.Figure) -> Image.Image:
    """Converte matplotlib Figure para PIL Image."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0E1117")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def make_gauge(score: float) -> Image.Image:
    """Cria um medidor circular (gauge) para o score."""
    fig, ax = plt.subplots(figsize=(3, 2.2), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    # Arco de fundo
    theta_bg = np.linspace(np.pi, 0, 100)
    ax.plot(theta_bg, [1] * 100, color="#333333", linewidth=18, solid_capstyle="round")

    # Arco do score
    pct = score / 100.0
    theta_fg = np.linspace(np.pi, np.pi * (1 - pct), max(int(100 * pct), 2))
    color = "#22c55e" if score < 40 else "#eab308" if score < 70 else "#ef4444"
    ax.plot(theta_fg, [1] * len(theta_fg), color=color, linewidth=18, solid_capstyle="round")

    # Texto central
    ax.text(
        np.pi / 2, 0.3, f"{score:.0f}%",
        ha="center", va="center", fontsize=26, fontweight="bold", color="white",
    )
    ax.text(
        np.pi / 2, -0.15, "DEEPFAKE SCORE",
        ha="center", va="center", fontsize=8, color="#999999",
    )

    ax.set_ylim(0, 1.4)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)

    return fig_to_image(fig)


def make_fft_plot(image_bgr: np.ndarray) -> Image.Image:
    """Gera plot do espectro de frequência."""
    spectrum = compute_spectrum(image_bgr)
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")
    ax.imshow(spectrum, cmap="inferno")
    ax.set_title("Espectro de Frequência (FFT 2D)", color="white", fontsize=10)
    ax.axis("off")
    return fig_to_image(fig)


# --------------------------------------------------------------------- #
# Configuração da página
# --------------------------------------------------------------------- #
st.set_page_config(
    page_title="DeepShield — AI Deepfake Detector",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stMetric > div { background: #1a1a2e; border-radius: 10px; padding: 12px; }
    h1 { text-align: center; }
    .verdict-real {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #22c55e; border-radius: 12px;
        padding: 20px; text-align: center; margin: 10px 0;
    }
    .verdict-fake {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #ef4444; border-radius: 12px;
        padding: 20px; text-align: center; margin: 10px 0;
    }
    .score-bar { border-radius: 6px; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## Configurações")
    threshold = st.slider("Threshold (fake se >= x%)", 10, 90, 50, 5)

    st.markdown("### Pesos do Ensemble")
    w_cnn = st.slider("CNN (EfficientNet)", 0.0, 1.0, 0.60, 0.05)
    w_fft = st.slider("Frequência (FFT)", 0.0, 1.0, 0.25, 0.05)
    w_cons = st.slider("Consistência", 0.0, 1.0, 0.15, 0.05)

    # Normaliza para somar 1
    w_total = w_cnn + w_fft + w_cons
    if w_total > 0:
        w_cnn, w_fft, w_cons = w_cnn / w_total, w_fft / w_total, w_cons / w_total
    st.caption(f"Normalizado: {w_cnn:.2f} / {w_fft:.2f} / {w_cons:.2f}")

    st.markdown("---")
    st.markdown("### Sobre")
    st.caption(
        "DeepShield usa EfficientNet-B0 + análise de frequência + "
        "detecção de artefatos para identificar imagens manipuladas por IA."
    )
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    st.caption(f"Device: {device_name}")
    st.caption(f"Modelo: {'Carregado' if WEIGHTS_PATH.exists() else 'Não encontrado'}")


# --------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------- #
st.markdown("# DeepShield — AI Deepfake Detector")
st.caption("Envie uma imagem para detectar se foi gerada ou manipulada por IA")

# --------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------- #
uploaded = st.file_uploader(
    "Arraste ou selecione uma imagem",
    type=["jpg", "jpeg", "png", "webp"],
    help="Formatos aceitos: JPG, PNG, WEBP",
)

if uploaded is not None:
    # Salva temporariamente para que o OpenCV consiga ler
    tmp_path = ROOT / "app" / f"_tmp_{uploaded.name}"
    tmp_path.write_bytes(uploaded.getvalue())

    try:
        model, device = load_model()

        with st.spinner("Analisando imagem..."):
            result: AnalysisResult = analyze_image(
                tmp_path, model, device,
                weights=(w_cnn, w_fft, w_cons),
            )

        # Aplica threshold customizado
        prediction = "fake" if result.deepfake_score >= threshold else "real"

        # --------------------------------------------------------------- #
        # Veredicto
        # --------------------------------------------------------------- #
        if prediction == "real":
            st.markdown(
                '<div class="verdict-real">'
                '<h2 style="color:#22c55e;margin:0;">Provavelmente Real</h2>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="verdict-fake">'
                '<h2 style="color:#ef4444;margin:0;">Provavelmente Fake</h2>'
                '</div>',
                unsafe_allow_html=True,
            )

        # --------------------------------------------------------------- #
        # Layout: Imagem + Grad-CAM | Gauge + Scores
        # --------------------------------------------------------------- #
        col_img, col_score = st.columns([3, 2])

        with col_img:
            img_col, cam_col = st.columns(2)
            pil_img = Image.open(uploaded)
            with img_col:
                st.markdown("**Imagem Original**")
                st.image(pil_img, use_container_width=True)
            with cam_col:
                st.markdown("**Grad-CAM — Regiões analisadas**")
                st.image(bgr_to_rgb(result.overlay), use_container_width=True)

        with col_score:
            # Gauge
            gauge_img = make_gauge(result.deepfake_score)
            st.image(gauge_img, use_container_width=True)

            # Sub-scores em métricas
            m1, m2, m3 = st.columns(3)
            m1.metric("CNN", f"{result.cnn_score:.0%}")
            m2.metric("FFT", f"{result.fft_score:.0%}")
            m3.metric("Consist.", f"{result.consistency_score:.0%}")

            # Barras visuais
            st.markdown("**Breakdown**")

            st.caption(f"CNN (EfficientNet) — {result.cnn_score:.1%}")
            st.progress(min(result.cnn_score, 1.0))

            st.caption(f"Frequência (FFT) — {result.fft_score:.1%}")
            st.progress(min(result.fft_score, 1.0))

            st.caption(f"Consistência — {result.consistency_score:.1%}")
            st.progress(min(result.consistency_score, 1.0))

        # --------------------------------------------------------------- #
        # Detalhes técnicos (expansível)
        # --------------------------------------------------------------- #
        with st.expander("Detalhes Técnicos — Análise de Frequência"):
            det_col1, det_col2 = st.columns(2)
            img_bgr = cv2.imread(str(tmp_path))
            with det_col1:
                st.markdown("**Espectro de Frequência (FFT 2D)**")
                fft_img = make_fft_plot(img_bgr)
                st.image(fft_img, use_container_width=True)
            with det_col2:
                st.markdown("**Interpretação**")
                st.markdown(
                    "O espectro mostra a distribuição de frequências espaciais. "
                    "Deepfakes gerados por GANs/diffusion tendem a apresentar:\n"
                    "- Atenuação nas altas frequências (imagem \"suavizada\")\n"
                    "- Picos periódicos (grid artifacts de upsampling)\n"
                    "- Padrões de simetria anômalos\n\n"
                    f"**FFT Score: {result.fft_score:.1%}** — "
                    f"{'indica anomalia no espectro' if result.fft_score > 0.4 else 'espectro dentro do esperado'}."
                )
                st.markdown("**Parâmetros da análise**")
                st.json({
                    "ensemble_weights": {"cnn": round(w_cnn, 2), "fft": round(w_fft, 2), "consistency": round(w_cons, 2)},
                    "threshold": threshold,
                    "deepfake_score": round(result.deepfake_score, 2),
                    "sub_scores": {
                        "cnn": round(result.cnn_score, 4),
                        "fft": round(result.fft_score, 4),
                        "consistency": round(result.consistency_score, 4),
                    },
                })

        # --------------------------------------------------------------- #
        # Como funciona
        # --------------------------------------------------------------- #
        with st.expander("Como funciona?"):
            t1, t2, t3 = st.columns(3)
            with t1:
                st.markdown("### CNN")
                st.markdown(
                    "Usa **EfficientNet-B0** treinado em 100k faces reais e "
                    "falsas. A rede aprendeu padrões visuais sutis que "
                    "diferenciam rostos reais de gerados por IA.\n\n"
                    "**Peso padrão: 60%**"
                )
            with t2:
                st.markdown("### Frequência")
                st.markdown(
                    "Aplica **FFT 2D** (Transformada de Fourier) para analisar "
                    "o espectro de frequência. GANs deixam assinaturas "
                    "características nas altas frequências.\n\n"
                    "**Peso padrão: 25%**"
                )
            with t3:
                st.markdown("### Consistência")
                st.markdown(
                    "Detecta **artefatos de edição**: bordas inconsistentes "
                    "(Canny), foco irregular (Laplaciano) e regiões "
                    "recomprimidas (Error Level Analysis).\n\n"
                    "**Peso padrão: 15%**"
                )

    finally:
        # Remove arquivo temporário
        if tmp_path.exists():
            tmp_path.unlink()

else:
    # Estado inicial — placeholder
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;padding:60px 20px;color:#666;">'
        '<p style="font-size:48px;margin-bottom:10px;">&#128752;</p>'
        "<h3>Nenhuma imagem carregada</h3>"
        "<p>Arraste uma imagem acima para iniciar a análise</p>"
        "</div>",
        unsafe_allow_html=True,
    )
