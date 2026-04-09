# DeepShield

Detector de deepfakes baseado em EfficientNet-B0 e PyTorch.

## Estrutura

- `src/` - código-fonte (modelo, dataset, treino, inferência)
- `app/` - interface Streamlit
- `models/` - pesos treinados (.pth)
- `data/` - datasets (raw/processed)
- `notebooks/` - experimentos
- `tests/` - testes com pytest

## Uso

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
