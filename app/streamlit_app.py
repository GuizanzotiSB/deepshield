"""Interface web do DeepShield usando Streamlit.

Permite ao usuário enviar uma imagem e receber a predição
de autenticidade (real ou deepfake).
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from src.predict import predict_image


def main() -> None:
    """Executa a aplicação Streamlit."""
    st.title("DeepShield - Detector de Deepfakes")
    st.write("Envie uma imagem para verificar se é real ou falsa.")

    uploaded = st.file_uploader("Imagem", type=["jpg", "jpeg", "png"])
    weights = Path("models/deepshield.pth")

    if uploaded is not None and weights.exists():
        image = Image.open(uploaded)
        st.image(image, caption="Imagem enviada", use_column_width=True)
        label, confidence = predict_image(uploaded, weights)
        st.metric("Resultado", label.upper(), f"{confidence:.2%}")


if __name__ == "__main__":
    main()
