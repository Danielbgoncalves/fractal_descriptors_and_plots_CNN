# Visão geral para LLMs — Projeto de descritores fractais

Objetivo: Fornecer uma visão concisa e estruturada do repositório, com uma árvore de relacionamento entre arquivos e o que fazem.

Resumo de execução
- Entrada exemplar: `entradas_para_testes/` (pastas com imagens/recorrências de teste)
- Extração de features fractais: `extract_fractal_features/` → gera CSVs de features
- Geração de imagens a partir de features/plots: `feature_to_image/` → imagens/plots utilizados por validações e modelos
- Treino/inferência: `model/` → dataset, modelo, treino, métricas
- Validações e comparações: `validacoes/` → scripts para avaliar/validar saídas
- Orquestrador: `main_pipeline.py` (ponto de entrada end-to-end)

Árvore de arquivos (nível alto)
- main_pipeline.py — Orquestra execução completa (extração → imagens → treino/validação)
- README.md — Documentação humana do projeto
- entradas_para_testes/
  - com_subpastas/ — estrutura de exemplo com subpastas (a1, a2/b1)
  - small/ — conjunto pequeno para testes rápidos

- extract_fractal_features/
  - clustperc.py, clustpercEucl.py, clustpercManh.py — cálculo de porcentagens/cluster usando diferentes distâncias
  - pmr.py, pmrEucl.py, pmrManh.py — funções PMR (medidas relacionadas a recorrência) por métrica
  - lacunaridade.py — cálculo de lacunaridade
  - D.py, N.py — medidas fractais (dimensão, contagens)
  - SaveCSVPercCLACDF3Distances.py, ScriptLACDF3Distances.py, ScriptPerc.py, ScriptPercLACDF3Distances.py — scripts para calcular e salvar CSVs combinando medidas e distâncias
  - util.py — utilitários usados pelos módulos acima
  - papel no fluxo: transformam imagens/plots em vetores de features e CSVs, usados pelo `model/` ou por `feature_to_image/`

- feature_to_image/
  - create_imgs.py — cria imagens a partir de features (output visual)
  - create_recorrence_plot.py — gera plots de recorrência a partir de dados
  - reshapeClassical.py, reshapeRecPlot.py — transformações/reshape de matrizes/imagens para modelo ou visualização
  - utils.py — utilitários locais
  - papel no fluxo: consome CSVs ou matrizes e gera imagens/plots para análise visual e validação

- model/
  - dataset.py — Dataset/loader que consome CSVs/imagens e prepara batches
  - model.py — definição do modelo (arquitetura usada para treino/inferência)
  - train.py — rotina de treino (loop, checkpointing)
  - run.py — rotina de execução/inferência (usar modelo treinado)
  - metrics.py — funções para cálculo de métricas de desempenho
  - utils.py — helpers do lado do modelo
  - papel no fluxo: recebe features/imagens, treina e gera métricas; `run.py` faz inferência

- validacoes/
  - otimizacao_em_relacao_ao_matlab/ — notas e experiência sobre otimização (documentação)
  - valida_csvs_de_features/
    - validacao.py — verificações/validações estatísticas de CSVs gerados
  - validacao_geracao_de_imagens/
    - compara_plots_gerados.py — compara plots gerados com referências; `resultado_comparacao.csv` com resultados
  - papel no fluxo: scripts que avaliam qualidade dos dados gerados e comparam com referências

Fluxo de dados e dependências (simplificado)
1. Entradas: `entradas_para_testes/` → fontes (imagens/matrizes)
2. `extract_fractal_features/` processa entradas → gera CSVs de features e/ou dados intermediários
3. `feature_to_image/` (opcional) converte features/recorrências → imagens/plots para inspeção e uso por modelos
4. `model/` consome CSVs ou imagens → `dataset.py` → `train.py` / `run.py` → produz métricas via `metrics.py`
5. `validacoes/` avalia CSVs e imagens geradas, compara com referências
6. `main_pipeline.py` junta etapas 2–5 em execução orquestrada

Entrypoints e comandos rápidos
- Rodar pipeline completo:
  - `python main_pipeline.py`
- Treinar modelo:
  - `python model/train.py`
- Inferir/rodar modelo treinado:
  - `python model/run.py`
- Validar CSVs:
  - `python validacoes/valida_csvs_de_features/validacao.py`

Dicas para LLMs que irão trabalhar com o código
- Comece por `extract_fractal_features/util.py` e `feature_to_image/utils.py` para entender formatos de dados esperados.
- `model/dataset.py` mostra como os CSVs/imagens devem ser organizados para treino.
- Scripts com prefixo `Script` e `SaveCSV` normalmente produzem artefatos (CSV) usados por outras etapas — procure por leitura/escrita em disco para rastrear I/O.
- Para mudanças de pré-processamento, ajuste `reshapeClassical.py` / `reshapeRecPlot.py` e valide via `validacoes/`.

Licença de leitura rápida
- Arquitetura: modular (features → imagens → modelo → validação)
- Pontos de integração: CSVs gerados em `extract_fractal_features/` são a interface principal entre extração e modelagem
- Arquivo de referência para começar: `main_pipeline.py`

Fim 
