# Visão geral do módulo XAI do projeto

Este diretório reúne a camada de explicabilidade do projeto: gera mapas de atenção/atribuição, compara arquiteturas e datasets, seleciona casos difíceis de decisão, agrega variação entre seeds e calcula métricas quantitativas de explicabilidade.

O objetivo geral não é apenas plotar um heatmap. O projeto usa os mapas de atribuição para responder perguntas científicas do tipo:

- Onde o modelo está olhando para classificar?
- O mapa muda perto da fronteira de decisão?
- Diferentes arquiteturas olham para regiões diferentes?
- Alguns descritores do RecPlot são mais importantes do que outros?
- A explicação continua plausível quando os pesos são aleatorizados?
- O ensemble corrige casos que um branch individual erra?

A arquitetura de alto nível é:

1. Modelos treinados em múltiplas seeds e branches são carregados.
2. Cada amostra passa por um método de explicabilidade (Grad-CAM, Integrated Gradients ou Occlusion).
3. O mapa é convertido para uma representação útil: 2D, normalizado, agregado por seed ou comparado entre arquiteturas.
4. As saídas entram em CSVs, análises por limiar e métricas de XAI.

---

## 1. Pipeline principal do projeto

O projeto tem uma cadeia de trabalho que começa no treinamento e nas predições e depois entra em XAI.

### 1.1 Input e estrutura de dados

Os dados de teste são carregados por `model/dataset.py`, em especial `EnsembleTestDataset`.

Esse dataset emparelha:

- imagem original
- imagem RecPlot
- label real

Ele retorna, para cada índice:

- `img_orig`: tensor normalizado da imagem original
- `img_rec`: tensor normalizado do RecPlot
- `label`: classe verdadeira

Essa estrutura é central porque a maioria do XAI decide qual entrada usar dependendo do branch:

- branch `originais` usa `img_orig`
- branch `recplot` usa `img_rec`

### 1.2 Geração das predições

`model/metrics.py` é o módulo que monta o CSV final de predições.

Ele:

- carrega os 4 branches por seed
- faz forward nos modelos para cada amostra
- combina as probabilidades em 6 cenários de ensemble
- salva as colunas `seed`, `cenario`, `sample`, `y_true`, `y_pred`, `prob_0`, `prob_1`, ...

Esses CSVs são a base de quase toda análise do XAI, principalmente em:

- `xai/case_selection.py`
- `xai/threshold_analysis.py`
- `xai/comparacao_arquiteturas.py`

---

## 2. Módulos do diretório XAI e sua função

## 2.1 xai/loader.py

Arquivo central de carregamento dos modelos.

### Funções

#### `carregar_todos_os_branches(seed, num_classes, models_dir="models", device="cuda")`

- Papel: abrir todos os 4 modelos treinados de uma seed específica.
- Entrada:
  - `seed`: identificador da semente
  - `num_classes`: número de classes do problema
  - `models_dir`: pasta onde os pesos estão salvos
  - `device`: CPU ou CUDA
- Saída:
  - `branches`: dicionário tipo
    - `mobilenet_orig`
    - `mobilenet_recplot`
    - `effnet_orig`
    - `effnet_recplot`
  - cada valor é o modelo carregado e em `eval()`
- Observação: se o arquivo não existir, ele deixa `None` e avisa.

#### `obter_camada_alvo(modelo)`

- Papel: localizar a última camada convolucional do backbone.
- Saída: `modelo.features[-1]`
- Uso: no Grad-CAM, o método precisa de uma camada alvo final para recuperar gradientes e mapas de ativação.

### Estrutura de nomes

`BRANCH_FILES` define o mapeamento entre nome interno e arquivo de pesos.

Exemplos:

- `mobilenet_orig` -> `mobilenet_originais.pth`
- `mobilenet_recplot` -> `mobilenet_RP_perc.pth`
- `effnet_orig` -> `efficientnet_b0_originais.pth`
- `effnet_recplot` -> `efficientnet_b0_RP_perc.pth`

---

## 2.2 xai/attributions.py

É o núcleo da explicabilidade. Aqui a imagem vira um mapa de importância.

### Funções

#### `desnormalizar(tensor_img, tipo)`

- Papel: reverter a normalização do tensor para a faixa visual do pixel, convertendo `[C, H, W]` para `H x W x C` em `[0,1]`.
- Entrada:
  - `tensor_img`: tensor de imagem já normalizado
  - `tipo`: `"orig"` ou `"rec"`
- Saída:
  - `np.ndarray` em formato de imagem exibível
- O que significa:
  - `orig` usa a normalização ImageNet
  - `rec` usa normalização específica do RecPlot, com média/desvio 0.5
- Usuário final: usado no plot de heatmap sobre a imagem original.

#### `gerar_atribuicao(modelo, input_tensor, target_class, camada_alvo=None, metodo="gradcam")`

- Papel: produzir o mapa de atribuição para um input e uma classe alvo.
- Entrada:
  - `modelo`: rede treinada
  - `input_tensor`: imagem tensor normalizada
  - `target_class`: classe que se quer explicar (normalmente o `y_pred` ou `y_true`)
  - `camada_alvo`: camada da rede para Grad-CAM
  - `metodo`: `"gradcam"`, `"ig"` ou `"occlusion"`
- Saída:
  - `torch.Tensor` com o mapa de atribuição
- Como funciona:

  - Grad-CAM:
    - usa `LayerGradCam(modelo, camada_alvo)`
    - calcula gradientes da classe alvo até a camada alvo
    - retorna mapa de relevância espacial
    - depois faz interpolação para o tamanho da imagem

  - Integrated Gradients:
    - usa `IntegratedGradients(modelo)`
    - integra o gradiente ao longo de um caminho de interpolação entre fundo e imagem

  - Occlusion:
    - usa `Occlusion(modelo)`
    - percorre janelas 3x16x16 em passos de 8 com impacto local
    - mede como a predição muda ao ocluir porções da imagem

- Observação importante:
  - a função faz `input_tensor.requires_grad_(True)` e usa a rede em modo de cálculo de gradiente
  - o retorno nem sempre é direto 2D; em vários pontos o código depois reduz o canal, por exemplo `mapa.mean(axis=0)`

---

## 2.3 xai/loader e attributions: o fluxo mínimo de explicação

O fluxo típico é:

1. escolher modelo e entrada apropriada
2. escolher camada alvo para Grad-CAM
3. chamar `gerar_atribuicao(...)`
4. reduzir o canal, transformar em 2D e normalizar para visualização

Exemplo de lógica típica do projeto:

```
camada = obter_camada_alvo(modelo)
attr = gerar_atribuicao(modelo, input_tensor, target_class, camada_alvo=camada, metodo="gradcam")
mapa = attr.detach().cpu().numpy()[0]
mapa_2d = mapa.mean(axis=0)
```

Isso significa: o mapa do Grad-CAM vem em `(1, C, H, W)` ou `(1, H, W)`, e o projeto reduz os canais para um único heatmap 2D.

---

## 2.4 xai/case_selection.py

Esse módulo responde a pergunta: “quais amostras merecem atenção por estarem em regiões de dúvida, conflito ou erro confiante?”

Ele usa o CSV de predições e calcula duas colunas derivadas:

- `confianca`: probabilidade atribuída à classe prevista
- `margem`: diferença entre a maior e a segunda maior probabilidade

### Funções

#### `_colunas_de_probabilidade(df)`

- Papel: descobrir colunas do tipo `prob_0`, `prob_1`, ...
- Retorno: lista ordenada das colunas de probabilidade

#### `_adiciona_confianca_e_margem(df)`

- Papel: criar colunas extra em um DataFrame de predição
- Saída:
  - `confianca`: probabilidade da classe prevista
  - `margem`: diferença entre maior e segunda maior probabilidade
- Significado:
  - baixa margem = modelo está indeciso
  - alta confiança = modelo está muito certo

#### `casos_corrigidos_pelo_ensemble(df, seed, cenario_individual, cenario_ensemble)`

- Papel: saber quais amostras um branch individual errou, mas o ensemble acertou.
- Retorno: lista de `sample` corrigidos
- Como funciona:
  - compara predições do cenário individual e do cenário ensemble
  - identifica `sample` em que o branch individual errou e o ensemble acertou

#### `casos_por_margem(df, seed, cenario, margem_max=0.1)`

- Papel: retornar os casos de fronteira do modelo.
- Retorno: DataFrame filtrado com `margem <= margem_max`
- Significado: são os casos em que o modelo quase não distingue a classe vencedora da segunda melhor

#### `casos_por_confianca(df, seed, cenario, tipo="acerto_confiante", n=5)`

- Papel: selecionar amostras por confiança
- Tipos válidos:
  - `acerto_confiante`: acertou e está muito confiante
  - `erro_confiante`: errou e estava muito confiante
  - `fronteira`: amostras com menor margem, independentemente de acerto
- Retorno: lista dos `sample`s mais relevantes

#### `casos_discordancia_entre_branches(df, seed, branch_a, branch_b)`

- Papel: encontrar amostras em que dois branches individuais discordam na predição
- Retorno: lista de amostras discordantes

#### `obter_amostras_do_dataset(dataset, indices)`

- Papel: transformar índices de sample em itens do dataset
- Retorno: lista de entradas do dataset

### Importância prática

Este módulo é o “filtro seletivo” de amostras para análise qualitativa. Ele alimenta `xai/threshold_analysis.py` para estudar:

- casos na fronteira de decisão
- casos certos mas muito confiantes
- casos errados mas muito confiantes

---

## 2.5 xai/seed_aggregation.py

Esse módulo trata a variabilidade entre sementes como fonte de informação. Em vez de olhar apenas uma seed, ele agrega mapas entre as 8 seeds do mesmo branch.

### Funções

#### `carrega_branch_todas_seeds(nome_branch, num_classes, seeds=None, models_dir="models", device="cuda")`

- Papel: carregar o mesmo branch em todas as seeds desejadas.
- Retorno: dicionário `seed -> modelo`

#### `empilhar_atribuicoes_entre_seeds(...)`

- Papel: gerar o mesmo mapa de atribuição para a mesma amostra em todas as seeds e empilhar em `(n_seeds, H, W)`.
- Retorno:
  - `pilha`: array com mapas empilhados
  - `seeds_usadas`: lista de seeds realmente carregadas
- Como funciona:
  - escolhe no branch correto a entrada correta
  - chama `gerar_atribuicao` para cada seed
  - reduz para 2D
  - empilha

#### `agregar_mapa_medio(pilha)`

- Papel: calcular mapa médio e desvio padrão entre as seeds.
- Retorno:
  - `media`: mapa médio de atribuição
  - `desvio`: variabilidade entre seeds
- Significado:
  - média = evidência consensual
  - desvio = incerteza ou instabilidade da explicação entre seeds

#### `normalizar_mapa(mapa, eps=1e-8)`

- Papel: normalizar min-max para intervalo `[0,1]`.
- Retorno: mapa normalizado

#### `mapa_agregado_para_amostra(...)`

- Papel: conveniência para uma amostra e um branch.
- Retorno:
  - `media`
  - `desvio`
  - `seeds_usadas`
- Essa função escolhe automaticamente o tipo de entrada (`originais` ou `recplot`) conforme o branch.

#### `seed_mais_representativa(resultados_teste_csv, cenario, ...)`

- Papel: escolher a seed mais próxima da média de performance do cenário.
- Retorno: número da seed
- Útil quando não se quer usar a média de mapas agregados porque fica muito espalhada.

#### `figura_mapa_agregado(media, desvio, img_fundo=None, titulo=None)`

- Papel: plotar lado a lado:
  - mapa médio
  - desvio-padrão entre seeds
- Retorno: `matplotlib.figure.Figure`

#### `salvar_mapas_agregados(media, desvio, seeds_usadas, output_dir, nome_base)`

- Papel: salvar os mapas e a lista de seeds em disco.
- Retorno: nada; salva arquivos `.npy` e `.csv`

### Interpretação filosófica

Este módulo responde à pergunta: “a explicação é estável ou depende da seed?”

Se o mapa médio é forte e o desvio é pequeno, a explicação é robusta. Se o desvio é alto, a explicação depende muito do treinamento inicial.

---

## 2.6 xai/threshold_analysis.py

Esse módulo analisa o comportamento do modelo perto do limiar de decisão, isto é, quando a probabilidade da classe vencedora está muito próxima da segunda maior.

### Funções

#### `_heatmap_unico(modelo, input_tensor, target_class, metodo="gradcam")`

- Papel: produzir um heatmap 2D para uma amostra isolada.
- Retorno: `np.ndarray` 2D
- Como funciona:
  - usa `gerar_atribuicao`
  - reduz `(C, H, W)` para um único plano espacial

#### `similaridade_ssim(mapa_a, mapa_b)`

- Papel: medir semelhança estrutural entre dois mapas.
- Retorno: valor em `[0,1]`
- O que significa:
  - `1.0` = mapas muito parecidos
  - `0.0` = mapas muito diferentes
- Usa `skimage.metrics.structural_similarity`

#### `ensemble_salvaria_amostra(df_pred, seed, nome_branch, sample)`

- Papel: responder se um ensemble salvaria uma predição do branch que errou.
- Retorno: `True/False`
- Lógica:
  - verifica se o branch faz parte de algum cenário de ensemble
  - verifica se esse sample foi corrigido pelo ensemble

#### `analisar_limiar_de_decisao(...)`

- Papel: rotina principal de análise do limiar.
- Entrada:
  - `predicoes_csv`: CSV de predições
  - `dataset_obj`: objeto dataset de teste
  - `nome_dataset`: `displasia` ou `pulmao`
  - `num_classes`
  - `output_dir`
  - `seeds`, `branches`, `margem_max`, `n_confiantes`, `models_dir`, `metodo`, `device`
- Retorno:
  - `df_resultado`: tabela detalhada por amostra e por grupo
  - `df_resumo`: tabela agregada por dataset x branch x categoria
- O que ele faz:
  1. lê predições
  2. para cada seed e branch,
  3. seleciona:
     - `fronteira`: casos cuja margem está abaixo do limiar
     - `acerto_confiante`: amostras certas e muito confiantes
     - `erro_confiante`: amostras erradas e muito confiantes
  4. gera heatmap para cada amostra relevante
  5. calcula complexidade do mapa via Quantus `Sparseness`
  6. compara cada mapa da fronteira com a média dos mapas de acerto confiante, usando SSIM
  7. marca se o ensemble salvaria a decisão em caso de erro
  8. salva dois CSVs no diretório de saída

#### `resumir_limiar(df)`

- Papel: agregar métricas por categoria.
- Retorno: DataFrame com médias e desvio de:
  - complexidade
  - SSIM em relação ao grupo confiante
  - acerto
  - porcentagem em que ensemble salvaria a decisão

#### `gerar_figura_exemplos(...)`

- Papel: gerar figura com exemplos visuais de 3 categorias por branch e seed.
- Retorno: `matplotlib.figure.Figure`
- O que mostra:
  - `acerto_confiante`
  - `fronteira`
  - `erro_confiante`
- Sobre cada amostra, mostra imagem + heatmap sobreposto

### Por que esse módulo é importante?

Ele transforma a explicabilidade em análise de decisão. Em vez de somente comentar “o heatmap ficou no centro”, ele mede se a explicação muda perto da fronteira de decisão, se o ensemble corrige a dúvida e se amostras próximas ao limiar têm mapas mais espalhados.

---

## 2.7 xai/xai_metrics.py

Este módulo calcula métricas quantitativas de qualidade das explicações usando a biblioteca `Quantus`.

### Funções

#### `_criar_explain_func(metodo="gradcam")`

- Papel: criar uma função compatível com o Quantus para gerar explicação por amostra em um batch.
- Retorno: função `explain` que recebe modelo, inputs e targets e devolve um batch de mapas
- Importância: Quantus exige essa função porque algumas métricas perturbam a entrada em várias iterações.

#### `calcula_matricas_xai(modelo, x_batch, y_batch, metodo="gradcam", ...)`

- Papel: calcular as métricas para um lote de amostras.
- Entrada:
  - `modelo`
  - `x_batch`: batch de imagens no formato do modelo
  - `y_batch`: labels alvo
  - `metodo`: `gradcam` etc.
- Saída:
  - dicionário com valores para:
    - `faithfulness`
    - `robustness`
    - `complexity`
- O que significa:

  - `faithfulness`: a região destacada realmente influencia a decisão?
  - `robustness`: pequenas perturbações na entrada mudam muito a explicação?
  - `complexity`: o mapa é concentrado ou espalhado?

- Internamente usa Quantus:
  - `FaithfulnessCorrelation`
  - `MaxSensitivity`
  - `Sparseness`

#### `xai_metrics_to_csv(...)`

- Papel: rodar as métricas para todas as seeds e branches e salvar em CSV.
- Entrada:
  - `seeds`
  - `num_classes`
  - `models_dir`
  - `dataset`
  - `output_dir`
  - `metodo`
  - `n_amostras`
  - `device`
  - parâmetros de Quantus
- Saída:
  - `df`: DataFrame de amostra/valor/metrica
  - salva em disco:
    - `xai_metricas.csv`
    - `xai_metricas_mean.csv`
- Estrutura do CSV final:
  - `seed`
  - `branch`
  - `amostra`
  - `metrica`
  - `valor`

#### `mostrar_metricas_xai(df)`

- Papel: imprimir resumo agrupado por branch e métrica
- Retorno: imprime `mean` e `std`

### Relevância

Este módulo é a parte “científica quantitativa” de XAI. Ele vai além de plotar heatmaps e busca um comparativo robusto entre métodos, branches e seeds.

---

## 2.8 xai/recplot_mapping.py

Esse módulo trata do RecPlot, que tem estrutura semântica por métricas e funções de percolação.

### Constantes importantes

- `TAMANHO_ORIGINAL_DESCRITORES = 180`
- `TAMANHO_REDE = 224`
- `RAIOS = list(range(3, 42, 2))`

Esses valores revelam o mapeamento entre o mapa 224x224 da CNN e o mapa de descritores 180x180 original.

### Funções

#### `pixel_to_descriptor(row, col, n_descritores=180, tamanho_final=224)`

- Papel: mapear um pixel do heatmap 224x224 para o descritor correspondente no RecPlot 180x180.
- Retorno:
  - `(desc_row, desc_col)` no espaço de descritores
  - `(info_row, info_col)` com metrica/função/raio associados
- Significado: esse é o “dicionário” para saber qual descritor foi mais relevante no espaço de features.

#### `decodificar_indice_descritor(idx)`

- Papel: transformar índice do descritor em informação semântica.
- Retorno: dicionário com:
  - `indice`
  - `metrica` (`Minkowski`, `Euclidiana`, `Manhattan`)
  - `funcao` (`p`, `g`, `h`)
  - `raio`
  - `label`

#### `definir_bandas_metricas()`

- Papel: mapear as 3 bandas principais do RecPlot por métrica.
- Retorno: dicionário de faixas

#### `definir_bandas_funcoes()`

- Papel: mapear sub-bandas por métrica+função (`P`, `G`, `H`)
- Retorno: 9 subbandas

#### `overlay_bandas_recplot(...)`

- Papel: desenhar o RecPlot com as bandas sobrepostas ao heatmap.
- Retorno: `matplotlib.figure.Figure`
- Útil para produzir visualização semântica da região relevante no RecPlot.

#### `avaliar_importancia_por_metrica_e_funcao(modelo, img_rec_tensor, target_class)`

- Papel: medir a importância quantitativa de cada sub-banda do RecPlot por oclusão.
- Retorno: dicionário nome_banda -> queda de probabilidade
- Como funciona:
  - para cada sub-banda, zera a região da imagem correspondente
  - recalcula a probabilidade da classe alvo
  - mede a queda de confiança `(prob_base - prob_oclusa)`

### O que esse módulo responde?

Ele funciona como a ponte entre o mapa de relevância visual e a semântica dos descritores. O projeto pode dizer:

- “a classe se sustenta principalmente em descritores Minkowski-p de raio 15”
- ou “as regiões `H` da métrica Manhattan são as mais importantes”

Esse módulo é fundamental para conectar XAI visual a análise de descritores.

---

## 2.9 xai/comparacao_arquiteturas.py

Esse arquivo compara a localização das evidências entre arquiteturas para a mesma amostra.

### Funções

#### `_normaliza_mapa(attr)`

- Papel: transformar um tensor de atribuição em mapa 2D normalizado
- Retorno: array 2D em `[0,1]`

#### `mapa_normalizado(modelo, input_tensor, target_class, metodo="gradcam")`

- Papel: encapsular a geração do mapa e a normalização
a
- Retorno: heatmap 2D normalizado

#### `iou_top_k(mapa_a, mapa_b, k_percentil=10.0)`

- Papel: medir sobreposição entre os `k%` pixels mais importantes.
- Retorno: valor de IoU em `[0,1]`
- O que significa:
  - `1.0` = as arquiteturas apontam quase exatamente para os mesmos pixels
  - `0.0` = não há sobreposição

#### `ssim_entre_mapas(mapa_a, mapa_b)`

- Papel: medir similaridade estrutural entre mapas.
- Retorno: SSIM em `[0,1]`
- More sensível a padrões espaciais e bordas do que IoU.

#### `comparar_arquiteturas_amostra(...)`

- Papel: comparar duas arquiteturas em uma única amostra.
- Retorno: dicionário com:
  - `sample`
  - `tipo_representacao`
  - `target_class`
  - `iou_top_k`
  - `ssim`
- Usa a mesma representação (`orig` ou `recplot`) para isolar arquitetura.

#### `comparar_arquiteturas_em_lotes(...)`

- Papel: aplicar a comparação para várias amostras.
- Retorno: DataFrame de várias linhas, uma por amostra

#### `cruzar_com_predicoes(df_comparacao, df_predicoes, seed, cenario_ensemble)`

- Papel: juntar o resultado da comparação com as predições do ensemble.
- Retorno: DataFrame cruzado com `acerto_ensemble`

#### `resumo_concordancia_vs_ensemble(df_cruzado)`

- Papel: avaliar se discordância espacial correlaciona com acerto/erro do ensemble.
- Retorno: média e desvio de IoU/SSIM por `acerto_ensemble`

### Pergunta científica que responde

Quando duas arquiteturas discordam sobre onde está a evidência, o ensemble passa a acertar mais ou menos?

---

## 2.10 xai/comparacao_datasets.py

Esse módulo compara a importância por banda entre datasets diferentes.

### Funções

#### `importancia_por_banda_amostra_todas_seeds(...)`

- Papel: para uma amostra e um branch RecPlot, medir a queda de probabilidade ao ocluir cada banda em cada seed.
- Retorno: DataFrame com colunas:
  - `seed`
  - `banda`
  - `queda_prob`

#### `importancia_por_banda_dataset(...)`

- Papel: agrupar essa análise para várias amostras de um dataset.
- Retorno: DataFrame de importância por `sample` + `target_class` + `banda`

#### `resumo_importancia_por_banda(df_importancia)`

- Papel: resumir a importância média por banda.
- Retorno: DataFrame com média, desvio e contagem por banda.

#### `comparar_importancia_entre_datasets(...)`

- Papel: comparar os rankings de importância entre dois datasets.
- Retorno: tabela com médias e diferença de rank

#### `figura_comparacao_bandas(...)`

- Papel: plotar gráfico de barras para comparação visual das bandas.
- Retorno: `matplotlib.figure.Figure`

### O que significa?

Esse módulo quantifica se as regiões semânticas mais importantes (Minkowski, Euclidiana, Manhattan, P/G/H) mudam conforme o dataset. Isso é muito forte para comparar `displasia` e `pulmao`, por exemplo.

---

## 2.11 xai/sanity_check.py

Esse módulo testa se o método de explicação faz sentido do ponto de vista causal.

### Funções

#### `_reinicializar_modulo(modulo)`

- Papel: resetar pesos de um submódulo em lugar.

#### `_blocos_do_backbone(modelo)`

- Papel: listar blocos do backbone em ordem descendente, do final para o início.

#### `_para_heatmap_2d(attr)`

- Papel: converter o tensor de atribuição em mapa 2D normalizado.

#### `teste_rand_em_cascata(modelo_treinado, input_tensor, target_class, metodo="gradcam", obter_camada=obter_camada_alvo)`

- Papel: randomizar os pesos do modelo em cascata e verificar se o heatmap muda de forma esperada.
- Retorno: lista de dicionários com:
  - `bloco`
  - `spearman`
  - `ssim`
- O que significa:
  - se o heatmap continua muito parecido mesmo quando o modelo é randomizado, então a explicação não está refletindo o aprendizado
  - se o mapa degrada progressivamente, a explicação está saneada

#### `rodar_sanity_check_todos_branches(branches, img_orig_tensor, img_rec_tensor, target_class, metodo="gradcam")`

- Papel: rodar o teste para todos os branches de uma vez.
- Retorno: dicionário `branch -> lista de resultados`

#### `plotar_curva_sanidade(...)`

- Papel: plotar a curva de similaridade ao longo da randomização e um traço por branch.
- Retorno: `Figure`

### Justificativa

Esse módulo é a validação mais importante de confiabilidade de um método de XAI. Ele segue a ideia de Adebayo: se a explicação sobreviver a destruição dos pesos aprendidos, ela provavelmente não é uma explicação real do modelo.

---

## 3. Como o projeto é chamado na prática

Existem vários pontos de entrada, mas os mais importantes são os scripts em `xai/teste/`.

### `xai/teste/teste_3.py`

Possivelmente testa uma etapa de análise com dados RecPlot e branches.

### `xai/teste/teste_4.py`

Possivelmente continua a análise de comparação entre modelos.

### `xai/teste/teste_6.py`

Disponibiliza pipeline de análise/visualização um pouco mais avançada.

### `xai/teste/teste_10.py`

É um exemplo concreto de uso de `threshold_analysis.py`.

Esse script:

- define dataset de teste
- carrega `predicoes.csv`
- chama `analisar_limiar_de_decisao(...)`
- salva CSVs em pasta de saída
- usa `gerar_figura_exemplos(...)` para criar figura dos casos de fronteira

Fluxo mínimo do exemplo:

```
from xai.threshold_analysis import analisar_limiar_de_decisao, gerar_figura_exemplos
...

df_resultado, df_resumo = analisar_limiar_de_decisao(
    predicoes_csv,
    test_dataset,
    nome_dataset,
    num_classes,
    output_dir,
    margem_max=0.1,
    n_confiantes=15,
    models_dir=models_dir,
    metodo="occlusion"
)
```

A saída fica em arquivos como:

- `limiar_analysis.csv`
- `limiar_analysis_summary.csv`

---

## 4. Estrutura dos CSVs mais relevantes

### `predicoes.csv`

Produzido por `model/metrics.py`.

Colunas principais:

- `seed`
- `cenario`
- `sample`
- `y_true`
- `y_pred`
- `prob_0`, `prob_1`, ...

Esse CSV é a base para quase tudo que segue.

### `xai_metricas.csv`

Produzido por `xai/xai_metrics.py`.

Colunas principais:

- `seed`
- `branch`
- `amostra`
- `metrica`
- `valor`

### `xai_metricas_mean.csv`

Agregado por branch e métrica, com `media` e `std`.

### `limiar_analysis.csv`

Produzido por `xai/threshold_analysis.py`.

Colunas principais:

- `dataset`
- `seed`
- `branch`
- `sample`
- `categoria`
- `confianca`
- `margem`
- `acertou`
- `complexidade_heatmap`
- `ssim_vs_grupo_confiante`
- `ensemble_salvaria`

### `limiar_analysis_summary.csv`

Agregado por `dataset`, `branch`, `categoria`.

---

## 5. Como os retornos são usados no código

A lógica geral do projeto é:

- predições -> selecionar amostras relevantes
- amostras relevantes -> gerar mapas de importância
- mapas -> métricas/visualizações/comparecimentos

### Exemplo 1: geração de heatmap para uma amostra

```
mapa = _heatmap_unico(modelo, input_tensor, y_pred, metodo="gradcam")
```

Retorno: heatmap 2D que pode ser sobreposto na imagem.

### Exemplo 2: seleção de amostras difíceis

```
df_fronteira = casos_por_margem(df_pred, seed, cenario, margem_max=0.1)
```

Retorno: amostras na fronteira da decisão.

### Exemplo 3: agregação entre seeds

```
media, desvio, seeds_usadas = mapa_agregado_para_amostra(...)
```

Retorno: mapa consensual e sua incerteza.

### Exemplo 4: qualidade de explicação

```
metricas = calcula_matricas_xai(modelo, x_batch, y_batch, metodo="gradcam")
```

Retorno: dicionário com `faithfulness`, `robustness` e `complexity`.

---

## 6. Resumo executivo da arquitetura

Em uma frase:

- `model/metrics.py` gera predições e probabilidade por cenário.
- `xai/case_selection.py` escolhe amostras relevantes.
- `xai/attributions.py` produz mapas de importância.
- `xai/seed_aggregation.py` mede estabilidade entre seeds.
- `xai/threshold_analysis.py` testa limiar e fronteira de decisão.
- `xai/xai_metrics.py` calcula métricas quantitativas de qualidade.
- `xai/recplot_mapping.py` conecta o heatmap ao significado dos descritores.
- `xai/comparacao_arquiteturas.py` compara arquiteturas.
- `xai/comparacao_datasets.py` compara datasets.
- `xai/sanity_check.py` valida se a explicação é confiável.

---

## 7. O que o projeto realmente entrega

O diretório XAI entrega um conjunto de respostas analíticas:

- onde o modelo está olhando
- se o padrão muda perto da fronteira de decisão
- se o ensemble corrige erros de branches individuais
- quanto uma explicação é confiável em termos estadísticos
- se diferentes arquiteturas focam em regiões distintas
- quais descritores do RecPlot são mais importantes
- se o método de XAI é robusto e não só um artefato visual

Isso faz do módulo XAI não apenas um pacote de visualização, mas uma infraestrutura de análise interpretável do comportamento do modelo em problemas de diagnóstico.

---

## 8. Recomendação de leitura

Para outra LLM entender rapidamente o escopo, a ordem recomendada de leitura é:

1. `model/dataset.py`
2. `model/metrics.py`
3. `xai/loader.py`
4. `xai/attributions.py`
5. `xai/case_selection.py`
6. `xai/threshold_analysis.py`
7. `xai/xai_metrics.py`
8. `xai/recplot_mapping.py`
9. `xai/seed_aggregation.py`
10. `xai/sanity_check.py`
11. `xai/comparacao_arquiteturas.py`
12. `xai/comparacao_datasets.py`

Essa sequência traz a lógica de cima para baixo: dados -> predição -> explicação -> análise qualitativa -> análise quantitativa -> interpretação semântica.
