**Visão Geral**
- **Descrição**: O diretório `model/` contém o código para treinar, avaliar e carregar modelos de classificação de imagens usados no projeto. O fluxo principal contempla preparação de datasets (originais e F-RecPlot), treinamento em K-Fold, salvamento dos melhores pesos por seed/backbone/tipo e avaliação de ensembles que combinam saídas de diferentes modelos.

**Estrutura de Arquivos**
- **`model/dataset.py`**: utilitários e `Dataset` usados para treino e teste.
- **`model/utils.py`**: constantes, transformações e utilitários (dispositivo, seeds, transform, parâmetros).
- **`model/model.py`**: funções para criar e carregar modelos (MobileNet, EfficientNet-B0).
- **`model/train.py`**: lógica de treino (K-Fold), `train_one_fold`, `run_kfold` e `train_seeds`.
- **`model/metrics.py`**: funções para calcular e agregar métricas e exportar `resultados_testes.csv`.
- **`model/run.py`**: script de exemplo que executa treinamento por seeds e avaliação de ensemble.
- **`model/USO.md`**: instruções de uso rápido (resumo já incluso neste README).

**Detalhes por Arquivo**

- **`model/dataset.py`**: [model/dataset.py](model/dataset.py)
  - **Funções/classes principais**:
    - `load_data_from_folders(dir_data, class_names, reshape_type)`:
      - **Entrada**: `dir_data` (path para `dataset_path`), `class_names` (lista de nomes de classes), `reshape_type` (string; ex.: `'originais'` ou `'F-RecPlot'`).
      - **Saída**: lista de tuplas `(caminho_imagem_str, class_index)` para todas as imagens encontradas em `dir_data/<class_name>/<reshape_type>/*`.
      - **Observações**: imprime avisos se a pasta de classe não existir.

    - `ImageDataset(Dataset)`:
      - **Entrada**: `data_list` (lista de `(path, label)`), `transform` (opcional, do `model/utils.py`).
      - **Saída**: implementa `__len__` e `__getitem__` retornando `(image, label)`, onde `image` é um `PIL.Image` transformado para tensor caso `transform` esteja definido.

    - `EnsembleTestDataset(Dataset)`:
      - **Entrada**: `root_dir`, `class_names`, `transform`.
      - **Operação**: para cada `class_name` procura `root_dir/<class_name>/originais` e `root_dir/<class_name>/F-RecPlot`. Emparelha arquivos por `stem` (nome sem extensão).
      - **Saída**: `__getitem__` retorna `(img_orig, img_rec, label)` (PIL ou tensors após `transform`).
      - **Extensões suportadas**: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`.

- **`model/utils.py`**: [model/utils.py](model/utils.py)
  - **Constantes**: `IMG_SIZE`, `BATCH_SIZE`, `EPOCHS`, `K_FOLDS`, `SEEDS`, `DEVICE`.
  - **Transform**: `transform` (resize, ToTensor, Normalize) — usar em datasets.
  - **Função**: `set_seed(seed)` — fixa RNG para `random`, `numpy`, `torch` (inclui configurações CUDA quando disponível).

- **`model/model.py`**: [model/model.py](model/model.py)
  - **Funções**:
    - `criar_modelo(backbone: str, num_classes: int, pretrained='DEFAULT')`:
      - **Entrada**: `backbone` (`'mobilenet'` ou `'efficientnet_b0'`), `num_classes`, `pretrained` (padrão `'DEFAULT'` para usar pesos do torchvision quando suportado).
      - **Saída**: `torch.nn.Module` com a camada final ajustada para `num_classes`.

    - `carregar_modelo(backbone, num_classes, path_weights)`:
      - **Entrada**: `path_weights` (arquivo `.pth` salvo previamente), `backbone`, `num_classes`.
      - **Saída**: modelo carregado em `DEVICE` e em modo `eval()`; retorna `None` se o arquivo não existir.

- **`model/train.py`**: [model/train.py](model/train.py)
  - **Funções principais**:
    - `train_one_fold(model, train_loader, val_loader, epochs=EPOCHS)`:
      - **Entrada**: `model` (torch module), `train_loader`, `val_loader`, `epochs`.
      - **Saída**: `(model, best_model_state, history, metrics)` onde `best_model_state` é `state_dict` do melhor fold, `history` contém listas de perda/accuracy por época e `metrics` contém informações do melhor epoch.

    - `run_kfold(dataset_path, dataset_type, class_names, backbone='mobilenet', seed=SEED)`:
      - **Entrada**: `dataset_path`, `dataset_type` (`'originais'` ou `'F-RecPlot'`), `class_names`, `backbone`, `seed`.
      - **Operação**: monta `data_list` via `load_data_from_folders`, aplica `StratifiedKFold` com `K_FOLDS`, treina `K_FOLDS` folds, salva o melhor modelo global em `models/{seed}/{backbone}_{dataset_type}.pth` e grava métricas/histórico em `results_kfold/{seed}/{dataset_type}/{backbone}`.
      - **Saída**: `best_model` (modelo com pesos do melhor `best_model_state` carregados).

    - `train_seeds(seeds, dataset_path, classes)`:
      - **Entrada**: `seeds` (lista), `dataset_path`, `classes`.
      - **Operação**: para cada seed chama `run_kfold` para as combinações (mobilenet/efficientnet × `F-RecPlot`/`originais`) e retorna um dicionário com modelos por seed.

- **`model/metrics.py`**: [model/metrics.py](model/metrics.py)
  - **Funções**:
    - `mostrar_metricas(nome_cenario, y_true, y_pred)`:
      - **Entrada**: arrays `y_true`, `y_pred`.
      - **Saída**: dicionário contendo `acc`, `f1_macro`, `recall_macro`, `specificity_macro`.

    - `metrics_to_csv(seeds, results, test_loader)`:
      - **Entrada**: `seeds` (lista), `results` (mapa de modelos por seed — chaves esperadas: `'mobnet_orig'`, `'mobnet_recplot'`, `'effnet_orig'`, `'effnet_recplot'`), `test_loader` (que fornece `imgs_orig, imgs_rec, labels`).
      - **Operação**: executa forward dos modelos (usando `imgs_orig` ou `imgs_rec` conforme mapeamento), soma probabilidades para formar ensembles em vários cenários, calcula métricas e salva `resultados_testes.csv`.

- **`model/run.py`**: [model/run.py](model/run.py)
  - **Papel**: script de exemplo que define caminhos (ex.: `displasia_dataset_path`, `displasia_test_dir`), constrói `EnsembleTestDataset`, chama `train_seeds` e carrega modelos via `carregar_modelo` para avaliar ensembles. Produz `resultados_testes.csv`.

**Fluxo de Treinamento (resumido)**
- Preparar pastas em `datasets/<dataset_name>/treino_e_validacao/<class_name>/<originais|F-RecPlot>/*`
- Chamar `train_seeds(SEEDS, dataset_path, classes)` (usado por `run.py`).
- `run_kfold` internamente chama `train_one_fold` por fold; o melhor `state_dict` global é salvo em `models/{seed}/{backbone}_{dataset_type}.pth`.

**Fluxo de Teste / Avaliação (resumido)**
- Preparar `datasets/<dataset_name>/teste/<class_name>/originais` e `.../F-RecPlot` com arquivos emparelhados por `stem`.
- Criar `EnsembleTestDataset(TEST_DIR, classes, transform)` e `DataLoader`.
- Carregar modelos salvos com `carregar_modelo` e usar `metrics_to_csv` para gerar `resultados_testes.csv`.

**Entradas e Saídas (resumo prático)**
- Entradas esperadas: estrutura de diretórios conforme `model/USO.md` e arquivos de imagem nas pastas `originais` e `F-RecPlot`.
- Saídas geradas automaticamente:
  - Modelos: `models/{seed}/{backbone}_{dataset_type}.pth`.
  - Métricas/Histórico K-Fold: `results_kfold/{seed}/{dataset_type}/{backbone}/metrics.csv` e `history.csv`.
  - Resultados de teste ensemble: `resultados_testes.csv`.

**Dependências principais**
- `torch`, `torchvision`, `Pillow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`.
- Use o `requirements.txt` do repositório para instalar versões compatíveis.

**Exemplo mínimo de execução**
- Rodar o script de avaliação (usa caminhos embutidos em `run.py`):

```bash
python model/run.py
```

**Boas práticas e dicas**
- Mantenha os nomes das pastas `originais` e `F-RecPlot` exatamente como no código.
- Ao preparar o dataset de teste, garanta pares com mesmo `stem` para cada imagem original e seu recplot correspondente.
- Se for reproduzir resultados, fixe seeds via `set_seed(seed)` e garanta que `DEVICE` está configurado corretamente (CUDA vs CPU).

