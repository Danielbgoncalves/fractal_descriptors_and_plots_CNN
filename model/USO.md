# Uso da pasta model — Estrutura de entrada e passo a passo

Este documento explica, de forma direta e prática, como organizar os dados e como o código em `model/` usa essas pastas. Serve como um lembrete rápido.

## 1) Visão geral rápida
- A pasta `model` contém o código para treinar modelos (k-fold) e testar ensembles que combinam redes treinadas em imagens "originais" e em plots chamados `F-RecPlot`.
- Para treinar: o código lê imagens dentro de `dataset_path/<classe>/<tipo>/` onde `<tipo>` é `originais` ou `F-RecPlot`.
- Para testar: o `EnsembleTestDataset` procura, por classe, duas subpastas com o mesmo nome-base de arquivo (stem): `originais` e `F-RecPlot`.

## 2) Estrutura esperada (treino / validação)
Exemplo (paths relativos):

```
datasets/
  dataset_displasia/
    treino_e_validacao/     <- este é `dataset_path` usado em `run.py`
      healthy/
        originais/
          img01.png
          img02.png
        F-RecPlot/
          img01.png
          img02.png
      severe/
        originais/
        F-RecPlot/
```

- O código usa `load_data_from_folders(dataset_path, class_names, reshape_type)` (veja `model/dataset.py`) para listar `dataset_path/<class_name>/<reshape_type>/*`.
- Portanto, para treinar com plots use `reshape_type='F-RecPlot'`; para treinar com as imagens normais use `reshape_type='originais'`.

## 3) Estrutura do conjunto de teste
Exemplo:

```
datasets/
  dataset_displasia/
    teste/                <- este é o `TEST_DIR` em `run.py`
      healthy/
        originais/
          img01.png
        F-RecPlot/
          img01.png
      severe/
        originais/
        F-RecPlot/
```

- Para o teste o código emparelha `originais` e `F-RecPlot` por `stem` do arquivo (nome sem extensão). Ex.: `img01.png` em `originais` deve ter `img01.png` em `F-RecPlot`.

## 4) Regras e convenções importantes
- Nomes de pastas: use exatamente `originais` e `F-RecPlot` (o código procura essas strings).
- Nome dos arquivos: os pares devem ter o mesmo nome-base (stem). Ex.: `IMG_001.png` (originais) <-> `IMG_001.png` (F-RecPlot).
- Extensões suportadas: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` (veja `valid_exts` em `model/dataset.py`).
- Se o script imprimir "Diretório ... não encontrado" verifique se o caminho e os nomes de classe/subpastas batem exatamente.
- Se o script imprimir "Par não encontrado para ..." significa que não existe o par na pasta `F-RecPlot` para o `originais` (ou vice-versa).

## 5) Fluxo de treinamento (intuitivo)
- Função principal: `train_seeds(SEEDS, dataset_path, classes)` em `model/train.py`.
- Para cada seed ela chama `run_kfold(dataset_path, dataset_type, classes, backbone, seed)` 4 vezes (mobilenet/eﬃcientnet × `F-RecPlot`/`originais`).
- `run_kfold` usa `load_data_from_folders(dataset_path, classes, dataset_type)` para montar a lista de imagens para o tipo escolhido e executa K-Fold (configurado em `model/utils.py`).
- O melhor modelo por seed/backbone/tipo é salvo em:

```
models/{seed}/{backbone}_{dataset_type}.pth
```

Ex.: `models/42/mobilenet_F-RecPlot.pth` e `models/42/mobilenet_originais.pth`.

## 6) Fluxo de teste / avaliação (intuitivo)
- `model/run.py` monta `EnsembleTestDataset(TEST_DIR, classes, transform)` que retorna tuplas `(img_orig, img_rec, label)` para cada par encontrado.
- O `run.py` carrega os modelos salvos (originais e F-RecPlot, MobileNet e EfficientNet) e passa os tensores apropriados para cada modelo.
- Em `model/metrics.py` os outputs de pares de modelos são somados para formar diferentes cenários de ensemble; os resultados são gravados em `resultados_testes.csv`.

## 7) Exemplos rápidos de comandos
- Ative seu ambiente virtual e, a partir do diretório raiz do projeto, rode:

```bash
python model/run.py
```

- O `run.py` usa os caminhos definidos no próprio arquivo (variáveis `displasia_dataset_path`, `lung_dataset_path`, `displasia_test_dir`, `lung_test_dir`) — edite essas variáveis se necessário.

## 8) Dicas de debug rápidas
- Verifique se há pastas vazias ou arquivos faltando; mensagens de aviso no console indicam exatamente qual pasta/arquivo está ausente.
- Se um modelo não for encontrado ao carregar, a função `carregar_modelo` imprime erro e retorna `None` — treine (ou ajuste os caminhos) antes de testar.
- Mantenha nomes de arquivos consistentes (mesmo stem e preferencialmente sem espaços).

