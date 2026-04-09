# DeepShield

Detector de Deepfakes.

## Stack

- **Linguagem:** Python 3.10+
- **Framework ML:** PyTorch
- **Framework Web:** Streamlit (MVP) → FastAPI + React (produção)
- **Modelo base:** EfficientNet-B0 com transfer learning
- **Dataset:** 140k Real and Fake Faces (Kaggle)

## Estrutura de pastas

```
src/         # código-fonte principal
models/      # pesos e definições de modelos
data/        # datasets e dados processados
notebooks/   # experimentos e análises
tests/       # testes com pytest
app/         # interface Streamlit / FastAPI
```

## Convenções

- Type hints obrigatórios em todas as funções e métodos
- Docstrings em português
- Testes escritos com `pytest`

## Regras

- **Não instalar pacotes sem avisar antes.**
- **Sempre rodar linting antes de finalizar qualquer tarefa.**
