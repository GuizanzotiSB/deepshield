"""Interface web do DeepShield — Design Premium.

Uso::

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import base64
import sys
from io import BytesIO
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ensemble import AnalysisResult, analyze_image  # noqa: E402
from src.frequency_analysis import compute_spectrum, compute_azimuthal_profile  # noqa: E402
from src.model import DeepShieldModel  # noqa: E402

VERSION = "1.0.0"
WEIGHTS_PATH = ROOT / "models" / "best_model.pth"


# ====================================================================== #
# Page config — DEVE ser a primeira chamada st.*
# ====================================================================== #
st.set_page_config(
    page_title="DeepShield — GZ",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ====================================================================== #
# Helpers
# ====================================================================== #
@st.cache_resource
def load_model() -> tuple[DeepShieldModel, torch.device]:
    """Carrega o modelo uma única vez e mantém em cache."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_weights = WEIGHTS_PATH.exists()
    # Se nao tem pesos treinados, usa pretrained=True para ter features do ImageNet
    model = DeepShieldModel(pretrained=(not has_weights)).to(device)
    if has_weights:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    return model, device


def pil_to_base64(img: Image.Image) -> str:
    """Converte PIL Image para string base64."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def np_to_base64(arr: np.ndarray) -> str:
    """Converte array BGR para base64 PNG."""
    _, encoded = cv2.imencode(".png", arr)
    return base64.b64encode(encoded.tobytes()).decode()


def fig_to_pil(fig: plt.Figure) -> Image.Image:
    """Converte matplotlib Figure para PIL Image."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#0a0a1a")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def make_gauge_svg(score: float) -> str:
    """Cria SVG de gauge semicircular."""
    if score < 35:
        color = "#22c55e"
    elif score < 65:
        color = "#eab308"
    else:
        color = "#ef4444"

    pct = max(score / 100.0, 0.01)
    r = 85
    cx, cy = 110, 100
    end_angle = np.pi * (1 - pct)
    ex = cx + r * np.cos(end_angle)
    ey = cy - r * np.sin(end_angle)
    large_arc = 1 if pct > 0.5 else 0

    svg = (
        '<div style="display:flex;justify-content:center;padding:15px 0;">'
        f'<svg width="220" height="140" viewBox="0 0 220 140">'
        '<defs>'
        '<filter id="gauge-glow"><feGaussianBlur stdDeviation="4" result="g"/>'
        '<feMerge><feMergeNode in="g"/><feMergeNode in="SourceGraphic"/></feMerge>'
        '</filter>'
        '</defs>'
        f'<path d="M {cx - r} {cy} A {r} {r} 0 1 1 {cx + r} {cy}" '
        'fill="none" stroke="rgba(255,255,255,0.04)" stroke-width="16" stroke-linecap="round"/>'
        f'<path d="M {cx - r} {cy} A {r} {r} 0 {large_arc} 1 {ex:.1f} {ey:.1f}" '
        f'fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round" '
        'filter="url(#gauge-glow)"/>'
        f'<text x="{cx}" y="{cy - 18}" text-anchor="middle" fill="white" '
        f'font-size="38" font-weight="800" font-family="Inter,system-ui">{score:.0f}%</text>'
        f'<text x="{cx}" y="{cy + 6}" text-anchor="middle" fill="{color}" '
        'font-size="10" font-family="Inter,system-ui" font-weight="600" '
        'letter-spacing="3" opacity="0.8">DEEPFAKE SCORE</text>'
        '</svg>'
        '</div>'
    )
    return svg


def make_score_card(icon: str, title: str, score: float, color: str, weight: str) -> str:
    """Gera HTML de um card de sub-score."""
    pct = min(score * 100, 100)
    return (
        '<div class="ds-score-card">'
        f'<div style="font-size:32px;margin-bottom:8px;">{icon}</div>'
        f'<div style="color:#94a3b8;font-size:11px;text-transform:uppercase;'
        f'letter-spacing:1.5px;font-weight:600;margin-bottom:6px;">{title}</div>'
        f'<div style="color:white;font-size:32px;font-weight:800;margin-bottom:12px;'
        f'line-height:1;">{score:.0%}</div>'
        '<div style="background:rgba(255,255,255,0.06);border-radius:8px;height:6px;'
        'overflow:hidden;margin-bottom:8px;">'
        f'<div style="width:{pct:.0f}%;height:100%;border-radius:8px;'
        f'background:{color};transition:width 1s ease;"></div>'
        '</div>'
        f'<div style="color:#475569;font-size:10px;font-weight:500;">Peso: {weight}</div>'
        '</div>'
    )


def make_fft_plot(image_bgr: np.ndarray) -> Image.Image:
    """Gera plot do espectro de frequência + perfil radial."""
    spectrum = compute_spectrum(image_bgr)
    profile = compute_azimuthal_profile(spectrum)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("#0a0a1a")

    ax1.set_facecolor("#0a0a1a")
    ax1.imshow(spectrum, cmap="inferno")
    ax1.set_title("Espectro de Potência (FFT 2D)", color="white", fontsize=11, pad=10)
    ax1.axis("off")

    ax2.set_facecolor("#0a0a1a")
    ax2.fill_between(range(len(profile)), profile, alpha=0.2, color="#7c3aed")
    ax2.plot(profile, color="#a78bfa", linewidth=1.5)
    ax2.set_title("Perfil Radial de Energia", color="white", fontsize=11, pad=10)
    ax2.set_xlabel("Frequência espacial", color="#64748b", fontsize=9)
    ax2.set_ylabel("Energia média", color="#64748b", fontsize=9)
    ax2.tick_params(colors="#475569", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color("#1e293b")
    ax2.grid(True, alpha=0.1, color="#475569")

    plt.tight_layout()
    return fig_to_pil(fig)


# ====================================================================== #
# CSS Global — SEM customização do file uploader
# ====================================================================== #
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', system-ui, sans-serif !important; }
.block-container { padding-top: 0.5rem !important; padding-bottom: 2rem !important; max-width: 1150px; }
.stApp { background: linear-gradient(170deg, #0a0a1a 0%, #0f0f2e 40%, #0a0a1a 100%) !important; }
#MainMenu, footer, header { visibility: hidden; }

.ds-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 24px;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
}
.ds-card:hover { border-color: rgba(124,58,237,0.2); box-shadow: 0 8px 30px rgba(124,58,237,0.08); transform: translateY(-2px); }

.ds-score-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 24px 18px; text-align: center;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
}
.ds-score-card:hover { border-color: rgba(124,58,237,0.3); box-shadow: 0 12px 40px rgba(124,58,237,0.12); transform: translateY(-4px); }

.ds-verdict { border-radius: 16px; padding: 24px 30px; text-align: center; }
.ds-verdict-real { background: linear-gradient(135deg, rgba(34,197,94,0.08), rgba(34,197,94,0.15)); border: 1px solid rgba(34,197,94,0.25); }
.ds-verdict-fake { background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(239,68,68,0.15)); border: 1px solid rgba(239,68,68,0.25); }

.ds-section-title { text-align: center; margin: 50px 0 30px; }
.ds-section-title h2 { font-size: 1.8rem; font-weight: 800; background: linear-gradient(135deg, #7c3aed, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 6px; }
.ds-section-title p { color: #64748b; font-size: 14px; }

.ds-metric { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 18px; text-align: center; transition: all 0.3s ease; }
.ds-metric:hover { border-color: rgba(124,58,237,0.2); transform: translateY(-2px); }
.ds-metric-value { font-size: 28px; font-weight: 800; color: white; }
.ds-metric-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }

section[data-testid="stSidebar"] { background: linear-gradient(180deg, #08081a, #0d0d28) !important; border-right: 1px solid rgba(255,255,255,0.04) !important; }
.streamlit-expanderHeader { font-weight: 600 !important; color: #e2e8f0 !important; }

.ds-footer { text-align: center; padding: 40px 0 15px; color: #334155; font-size: 12px; border-top: 1px solid rgba(255,255,255,0.04); margin-top: 40px; }
.ds-footer a { color: #7c3aed; text-decoration: none; }

/* Fix botão upload com texto sobreposto (bug i18n Streamlit) */
[data-testid="stFileUploader"] button {
    font-size: 0 !important;
    color: transparent !important;
}
[data-testid="stFileUploader"] button * {
    font-size: 0 !important;
    color: transparent !important;
}
[data-testid="stFileUploader"] button::after {
    content: "Escolher";
    font-size: 14px !important;
    color: #333 !important;
}
</style>""", unsafe_allow_html=True)


# ====================================================================== #
# Sidebar
# ====================================================================== #
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:20px 0 10px;">'
        '<svg width="40" height="40" viewBox="0 0 52 52">'
        '<defs><linearGradient id="gz-sb" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#7c3aed"/><stop offset="100%" stop-color="#3b82f6"/>'
        '</linearGradient></defs>'
        '<circle cx="26" cy="26" r="25" fill="url(#gz-sb)"/>'
        '<text x="26" y="27" text-anchor="middle" dominant-baseline="central" '
        'fill="white" font-size="18" font-weight="800" font-family="Inter,system-ui">GZ</text>'
        '</svg>'
        f'<div style="color:#94a3b8;font-size:11px;margin-top:6px;letter-spacing:1px;'
        f'text-transform:uppercase;font-weight:600;">DeepShield v{VERSION}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    threshold = st.slider(
        "Limiar de decisão", min_value=10, max_value=90, value=50, step=5,
        help="Imagens com score acima deste valor sao classificadas como deepfake.",
    )

    st.markdown("---")
    st.markdown("**Pesos do ensemble**")
    w_cnn = st.slider("CNN (EfficientNet)", 0.0, 1.0, 0.60, 0.05)
    w_fft = st.slider("Frequência (FFT)", 0.0, 1.0, 0.25, 0.05)
    w_cons = st.slider("Consistência", 0.0, 1.0, 0.15, 0.05)

    w_total = w_cnn + w_fft + w_cons
    if w_total > 0:
        w_cnn, w_fft, w_cons = w_cnn / w_total, w_fft / w_total, w_cons / w_total
    st.caption(f"Normalizado: {w_cnn:.0%} / {w_fft:.0%} / {w_cons:.0%}")

    st.markdown("---")
    device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    model_status = "Carregado" if WEIGHTS_PATH.exists() else "ImageNet (sem treino)"
    st.markdown(
        f'<div style="color:#475569;font-size:11px;line-height:2;">'
        f'<b style="color:#64748b;">Device:</b> {device_name}<br>'
        f'<b style="color:#64748b;">Modelo:</b> {model_status}<br>'
        f'<b style="color:#64748b;">Backbone:</b> EfficientNet-B0</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<a href="https://github.com/GuizanzotiSB/deepshield" target="_blank" '
        'style="display:block;text-align:center;margin-top:12px;padding:10px;'
        'border-radius:10px;background:rgba(255,255,255,0.03);'
        'border:1px solid rgba(255,255,255,0.06);color:#94a3b8;'
        'text-decoration:none;font-size:13px;font-weight:600;">Star no GitHub</a>',
        unsafe_allow_html=True,
    )


# ====================================================================== #
# Hero
# ====================================================================== #
st.markdown(
    '<div style="text-align:center;padding:25px 0 5px;">'
    '<div style="display:inline-flex;align-items:center;gap:16px;margin-bottom:8px;">'
    '<svg width="56" height="56" viewBox="0 0 56 56">'
    '<defs><linearGradient id="gz-main" x1="0%" y1="0%" x2="100%" y2="100%">'
    '<stop offset="0%" stop-color="#7c3aed"/><stop offset="50%" stop-color="#3b82f6"/>'
    '<stop offset="100%" stop-color="#ec4899"/></linearGradient></defs>'
    '<circle cx="28" cy="28" r="27" fill="url(#gz-main)"/>'
    '<text x="28" y="29" text-anchor="middle" dominant-baseline="central" '
    'fill="white" font-size="22" font-weight="800" font-family="Inter,system-ui" '
    'letter-spacing="0.5">GZ</text></svg>'
    '<span style="font-size:3rem;font-weight:900;letter-spacing:-1px;'
    'background:linear-gradient(135deg,#7c3aed,#3b82f6,#ec4899);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
    'DeepShield</span></div>'
    '<p style="color:#94a3b8;font-size:1.05rem;margin:0 0 16px;">'
    'Detecção de deepfakes com Inteligência Artificial</p>'
    '<div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap;">'
    '<span style="padding:5px 14px;border-radius:20px;font-size:12px;font-weight:600;'
    'background:rgba(34,197,94,0.1);color:#4ade80;border:1px solid rgba(34,197,94,0.2);">'
    '99.84% Accuracy</span>'
    '<span style="padding:5px 14px;border-radius:20px;font-size:12px;font-weight:600;'
    'background:rgba(124,58,237,0.1);color:#a78bfa;border:1px solid rgba(124,58,237,0.2);">'
    'EfficientNet-B0</span>'
    '<span style="padding:5px 14px;border-radius:20px;font-size:12px;font-weight:600;'
    'background:rgba(236,72,153,0.1);color:#f472b6;border:1px solid rgba(236,72,153,0.2);">'
    'Ensemble 3 técnicas</span></div></div>'
    '<div style="height:2px;background:linear-gradient(90deg,transparent,'
    'rgba(124,58,237,0.15),rgba(59,130,246,0.15),transparent);margin:20px 0 25px;"></div>',
    unsafe_allow_html=True,
)


# ====================================================================== #
# Upload — sem CSS customizado, aceita visual padrão do Streamlit
# ====================================================================== #
uploaded = st.file_uploader(
    "", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed",
)

if uploaded is None:
    st.markdown(
        '<div style="max-width:650px;margin:10px auto;padding:55px 20px;text-align:center;'
        'border-radius:20px;border:2px dashed rgba(124,58,237,0.2);'
        'background:linear-gradient(145deg,rgba(124,58,237,0.03),rgba(59,130,246,0.03));">'
        '<div style="font-size:52px;margin-bottom:12px;opacity:0.7;">&#128424;</div>'
        '<div style="color:#94a3b8;font-size:16px;font-weight:500;">'
        'Envie uma imagem para analisar</div>'
        '<div style="color:#475569;font-size:12px;margin-top:8px;">JPG, PNG, WEBP</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    _has_result = False
else:
    _has_result = True

# ====================================================================== #
# Resultados (só após upload)
# ====================================================================== #
if _has_result and uploaded is not None:
    tmp_path = ROOT / "app" / f"_tmp_{uploaded.name}"
    tmp_path.write_bytes(uploaded.getvalue())

    try:
        model, device = load_model()

        with st.spinner("Analisando imagem..."):
            result: AnalysisResult = analyze_image(
                tmp_path, model, device, weights=(w_cnn, w_fft, w_cons),
            )

        prediction = "fake" if result.deepfake_score >= threshold else "real"

        # ----- Veredicto ----- #
        if prediction == "real":
            st.markdown(
                '<div class="ds-verdict ds-verdict-real">'
                '<div style="font-size:36px;margin-bottom:4px;">&#9989;</div>'
                '<h2 style="color:#4ade80;font-size:1.6rem;font-weight:800;margin:0;">'
                'Provavelmente Real</h2>'
                f'<p style="color:#6ee7b7;font-size:14px;margin:6px 0 0;opacity:0.8;">'
                f'Confiança: {result.confidence_level} &bull; '
                f'Score: {result.deepfake_score:.1f}%</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="ds-verdict ds-verdict-fake">'
                '<div style="font-size:36px;margin-bottom:4px;">&#9888;&#65039;</div>'
                '<h2 style="color:#f87171;font-size:1.6rem;font-weight:800;margin:0;">'
                'Provavelmente Fake</h2>'
                f'<p style="color:#fca5a5;font-size:14px;margin:6px 0 0;opacity:0.8;">'
                f'Confiança: {result.confidence_level} &bull; '
                f'Score: {result.deepfake_score:.1f}%</p></div>',
                unsafe_allow_html=True,
            )

        # ----- Imagens: Original | Grad-CAM ----- #
        col_orig, col_cam = st.columns(2, gap="medium")
        pil_img = Image.open(uploaded)
        orig_b64 = pil_to_base64(pil_img)
        overlay_rgb = cv2.cvtColor(result.overlay, cv2.COLOR_BGR2RGB)
        overlay_b64 = np_to_base64(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

        with col_orig:
            st.markdown(
                '<div class="ds-card">'
                '<div style="color:#64748b;font-size:11px;text-transform:uppercase;'
                'letter-spacing:1.5px;font-weight:600;margin-bottom:12px;'
                'text-align:center;">Imagem Original</div>'
                f'<img src="data:image/png;base64,{orig_b64}" '
                'style="width:100%;border-radius:10px;display:block;"/></div>',
                unsafe_allow_html=True,
            )

        with col_cam:
            st.markdown(
                '<div class="ds-card">'
                '<div style="color:#64748b;font-size:11px;text-transform:uppercase;'
                'letter-spacing:1.5px;font-weight:600;margin-bottom:12px;'
                'text-align:center;">Mapa de Calor (Grad-CAM)</div>'
                f'<img src="data:image/png;base64,{overlay_b64}" '
                'style="width:100%;border-radius:10px;display:block;"/></div>',
                unsafe_allow_html=True,
            )

        # ----- Gauge + Score Cards ----- #
        g_col, cards_col = st.columns([2, 3], gap="large")

        with g_col:
            gauge_html = make_gauge_svg(result.deepfake_score)
            st.markdown(
                f'<div class="ds-card" style="display:flex;align-items:center;'
                f'justify-content:center;min-height:200px;">{gauge_html}</div>',
                unsafe_allow_html=True,
            )

        with cards_col:
            c1, c2, c3 = st.columns(3, gap="small")
            with c1:
                st.markdown(make_score_card(
                    "&#129504;", "CNN", result.cnn_score,
                    "#7c3aed", f"{w_cnn:.0%}",
                ), unsafe_allow_html=True)
            with c2:
                st.markdown(make_score_card(
                    "&#128225;", "Frequência", result.fft_score,
                    "#3b82f6", f"{w_fft:.0%}",
                ), unsafe_allow_html=True)
            with c3:
                st.markdown(make_score_card(
                    "&#128269;", "Consistência", result.consistency_score,
                    "#ec4899", f"{w_cons:.0%}",
                ), unsafe_allow_html=True)

        # ----- Análise de frequência (expansível) ----- #
        with st.expander("Análise de Frequência"):
            img_bgr = cv2.imread(str(tmp_path))
            fft_col, info_col = st.columns([3, 2], gap="medium")
            with fft_col:
                fft_img = make_fft_plot(img_bgr)
                st.image(fft_img, use_container_width=True)
            with info_col:
                fft_status = ("anomalia detectada" if result.fft_score > 0.4
                              else "dentro do esperado")
                st.markdown(
                    f"**FFT Score: {result.fft_score:.1%}** — _{fft_status}_\n\n"
                    "O espectro revela padrões invisíveis a olho nu.\n"
                    "Imagens geradas por IA apresentam:\n"
                    "- **Atenuação** nas altas frequências\n"
                    "- **Picos periódicos** (grid artifacts)\n"
                    "- **Simetria anômala** no espectro 2D"
                )
                st.json({
                    "deepfake_score": round(result.deepfake_score, 2),
                    "sub_scores": {
                        "cnn": round(result.cnn_score, 4),
                        "fft": round(result.fft_score, 4),
                        "consistency": round(result.consistency_score, 4),
                    },
                    "threshold": threshold,
                })

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ====================================================================== #
# Como funciona (sempre visível)
# ====================================================================== #
st.markdown(
    '<div class="ds-section-title">'
    '<h2>Como funciona</h2>'
    '<p>Três técnicas complementares combinadas em um ensemble inteligente</p>'
    '</div>',
    unsafe_allow_html=True,
)

hw1, hw2, hw3 = st.columns(3, gap="medium")
with hw1:
    st.markdown(
        '<div class="ds-card" style="text-align:center;min-height:280px;">'
        '<div style="font-size:42px;margin-bottom:12px;">&#129504;</div>'
        '<h3 style="color:white;font-size:18px;font-weight:700;margin-bottom:8px;">'
        'Rede Neural (CNN)</h3>'
        '<p style="color:#94a3b8;font-size:13px;line-height:1.7;">'
        '<b style="color:#a78bfa;">EfficientNet-B0</b> treinada em 100k faces '
        'reais e geradas por IA. Aprende padrões visuais sutis como textura '
        'de pele, reflexos nos olhos e consistência de iluminação.'
        '<br><br><span style="color:#7c3aed;font-weight:600;">Contribuição: 60%</span>'
        '</p></div>',
        unsafe_allow_html=True,
    )
with hw2:
    st.markdown(
        '<div class="ds-card" style="text-align:center;min-height:280px;">'
        '<div style="font-size:42px;margin-bottom:12px;">&#128225;</div>'
        '<h3 style="color:white;font-size:18px;font-weight:700;margin-bottom:8px;">'
        'Análise de Frequência</h3>'
        '<p style="color:#94a3b8;font-size:13px;line-height:1.7;">'
        '<b style="color:#60a5fa;">FFT 2D</b> (Transformada de Fourier) converte '
        'a imagem para o domínio de frequência. GANs e modelos de difusão deixam '
        'assinaturas espectrais únicas e detectáveis.'
        '<br><br><span style="color:#3b82f6;font-weight:600;">Contribuição: 25%</span>'
        '</p></div>',
        unsafe_allow_html=True,
    )
with hw3:
    st.markdown(
        '<div class="ds-card" style="text-align:center;min-height:280px;">'
        '<div style="font-size:42px;margin-bottom:12px;">&#128269;</div>'
        '<h3 style="color:white;font-size:18px;font-weight:700;margin-bottom:8px;">'
        'Consistência</h3>'
        '<p style="color:#94a3b8;font-size:13px;line-height:1.7;">'
        'Três verificações: <b style="color:#f472b6;">Laplaciano</b> (foco), '
        '<b style="color:#f472b6;">Canny</b> (bordas) e '
        '<b style="color:#f472b6;">ELA</b> (recompressão JPEG). '
        'Detecta artefatos de blending e costuras.'
        '<br><br><span style="color:#ec4899;font-weight:600;">Contribuição: 15%</span>'
        '</p></div>',
        unsafe_allow_html=True,
    )

# ====================================================================== #
# Sobre o modelo (sempre visível)
# ====================================================================== #
st.markdown(
    '<div class="ds-section-title">'
    '<h2>Sobre o Modelo</h2>'
    '<p>Treinado no dataset 140k Real and Fake Faces do Kaggle</p>'
    '</div>',
    unsafe_allow_html=True,
)

a1, a2, a3, a4, a5, a6 = st.columns(6, gap="small")
metrics_data = [
    ("99.84%", "Accuracy"), ("0.9984", "F1 Score"), ("99.85%", "Precision"),
    ("99.83%", "Recall"), ("100k", "Imagens"), ("5.3M", "Parâmetros"),
]
for col, (val, label) in zip([a1, a2, a3, a4, a5, a6], metrics_data):
    with col:
        st.markdown(
            f'<div class="ds-metric">'
            f'<div class="ds-metric-value">{val}</div>'
            f'<div class="ds-metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ====================================================================== #
# Footer
# ====================================================================== #
st.markdown(
    '<div class="ds-footer">'
    'Desenvolvido por <a href="https://github.com/GuizanzotiSB" target="_blank">'
    'Guilherme Zanzoti</a> &bull; '
    '<a href="https://github.com/GuizanzotiSB/deepshield" target="_blank">GitHub</a>'
    ' &bull; PyTorch + Streamlit</div>',
    unsafe_allow_html=True,
)
