# Visão geral do projeto

Este repositório fornece ferramentas para extração de características fractais a partir de imagens e um fluxo simples para salvar esses resultados e treinar/avaliar modelos com eles.

Como funciona
- O diretório [extract_fractal_features](extract_fractal_features) contém implementações e scripts que calculam medidas fractais (por exemplo, PMR, lacunaridade, D, N, clustperc) e scripts utilitários para exportar resultados.
- Scripts como `SaveCSVPercCLACDF3Distances.py`, `ScriptLACDF3Distances.py` e `ScriptPercLACDF3Distances.py` geram arquivos CSV com as features processadas.
- O diretório [model](model) contém o pipeline de modelagem: preparação de datasets (`dataset.py`), definição de modelos (`model.py`), treinamento (`train.py`) e execução/avaliação (`run.py`, `utils.py`).

Uso básico
- Extraia features executando os scripts em [extract_fractal_features](extract_fractal_features) para gerar os CSVs de saída.
- Treine e avalie modelos usando os scripts em [model](model) (por exemplo, `model/train.py` ou `model/run.py`).
- Requisitos: Python 3 e bibliotecas científicas comuns (por exemplo, `numpy`, `scipy`, `scikit-learn`).

Créditos
Projeto vinculado ao LIPAI — Laboratório Interdisciplinar de Processamento e Análise de Imagens da UNIFESP.
