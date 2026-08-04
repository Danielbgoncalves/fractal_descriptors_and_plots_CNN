# Guia de Implementação — XAI no Pipeline Atual (fractal_descriptors_and_plots_CNN, branch `unicamp`)

Este guia parte do código real do repositório (analisei `model/dataset.py`, `model.py`, `train.py`, `run.py`, `metrics.py`, `utils.py` e os CSVs de `resultados_displasia/` e `resultados_pulmao/`) e organiza a implementação do plano de XAI (`plano_xai_recplot_percolacao.md`) em etapas que se encaixam no que já existe, sem reescrever o que já funciona. Cada etapa referencia diretamente as classes/funções já implementadas.

---

## Etapa 0 — Ajustes antes de começar (higiene mínima, não é refatoração)

Antes de tocar em XAI, três coisas no código atual vão te causar dor de cabeça depois se não forem resolvidas agora:

1. **`EnsembleTestDataset.__init__` (`dataset.py`, linhas 56–63) não ordena os arquivos** (`path_rec.iterdir()` e `path_orig.iterdir()` sem `sorted()`). Isso significa que o índice `sample` do `predicoes.csv` só corresponde de forma confiável a um arquivo específico **dentro da mesma execução do script**. Se você rodar `run.py` de novo (mesmo sem mudar nada), a ordem do sistema de arquivos pode mudar e o `sample=0` de hoje pode não ser o mesmo `sample=0` de amanhã.
   → **Ação:** adicione `sorted(...)` nas duas listas (`orig_files` e a criação de `rec_dict`) antes de qualquer coisa. É uma linha, mas é pré-requisito para a Etapa 6 (estudos de caso), que depende de reabrir uma amostra específica pelo índice salvo no CSV.
2. **`metrics.py` tem um bloco duplicado** (a partir da linha ~205: `all_results = []` de novo, `cenarios` redefinido de novo, um segundo loop `for seed in seeds` que recalcula e **sobrescreve** `resultados_testes.csv`). Não é bug funcional grave (o resultado final é o mesmo), mas dobra o tempo de execução à toa e vai confundir quando você for adaptar essa lógica para gerar `xai_metrics.csv` no mesmo padrão. Vale limpar antes de copiar o padrão para o módulo novo.
3. **`trainable_blocks=7` é usado em `run_kfold`** (`train.py`, linha 176), não o default `3` de `criar_modelo`. Isso importa diretamente para a escolha da camada-alvo do Grad-CAM (Etapa 2): com 7 blocos finais destravados, uma fatia bem maior da rede foi ajustada aos seus dados — documentar esse valor no artigo, na seção de métodos.

Nenhuma dessas mudanças toca no pipeline de treino existente — são leituras/ordenações, não mudam pesos nem métricas já calculadas.

---

## Etapa 1 — Infra de carregamento para XAI (`model/xai/loader.py`)

**Estado atual:** `run.py` (linhas 39–44) já carrega os 4 branches de uma seed manualmente, chamando `carregar_modelo` de `model.py` com caminhos hardcoded `models/{seed}/{backbone}_{dataset_type}.pth`. Isso já é 90% do que a Etapa 1 do plano de XAI precisa — só falta generalizar.

**O que desenvolver:** um único helper que reaproveita `carregar_modelo` (não recriar):

```python
# model/xai/loader.py
from ..model import carregar_modelo

BRANCH_FILES = {
    "mobilenet_orig":     ("mobilenet",        "originais", "mobilenet_originais.pth"),
    "mobilenet_recplot":  ("mobilenet",        "recplot",   "mobilenet_RP_perc.pth"),
    "effnet_orig":        ("efficientnet_b0",  "originais", "efficientnet_b0_originais.pth"),
    "effnet_recplot":     ("efficientnet_b0",  "recplot",   "efficientnet_b0_RP_perc.pth"),
}

def carregar_todos_os_branches(seed, num_classes, models_dir="models"):
    branches = {}
    for nome, (backbone, tipo, arquivo) in BRANCH_FILES.items():
        caminho = f"{models_dir}/{seed}/{arquivo}"
        branches[nome] = carregar_modelo(backbone, num_classes, tipo, caminho)
    return branches
```

**Camada-alvo para Grad-CAM:** como `mobilenet_v2` e `efficientnet_b0` do torchvision, ambos usados via `model.features[...]` em `criar_modelo` (`model.py`, linhas 101–141), a camada final de convolução em ambos é `model.features[-1]`. Use essa mesma referência para os dois backbones — dá pra comparar MobileNet vs. EfficientNet no mesmo "nível" de representação (RQ2 do plano anterior).

```python
def camada_alvo(model):
    return model.features[-1]
```

**Ligação com o dataset:** para gerar explicações, reuse `EnsembleTestDataset` (já existe, já pareia original+RecPlot pelo nome do arquivo) — não precisa de dataset novo, só iterar sobre ele.

---

## Etapa 2 — Módulo de atribuição (`model/xai/attributions.py`)

**Estado atual:** você já usa Captum "cru" (não vi o código específico, mas foi descrito). O que falta é sistematizar para os 4 branches e cuidar de um detalhe fácil de esquecer:

**Cuidado concreto do seu pipeline:** `utils.py` define **duas normalizações diferentes**:
- `transform_originais`: `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]` (ImageNet)
- `transform_recplot`: `mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]`

Se você usar uma função única de "desnormalizar para visualizar o heatmap sobreposto", ela **precisa saber qual dos dois branches está processando** — usar o mean/std errado não quebra o código, mas gera uma imagem de fundo com cores erradas por baixo do heatmap, e isso é o tipo de erro que passa despercebido até alguém perguntar na banca por que o RecPlot "parece" ter cores estranhas.

```python
# model/xai/attributions.py
from captum.attr import IntegratedGradients, LayerGradCam, Occlusion

MEAN_STD = {
    "orig": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "rec":  ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
}

def desnormalizar(tensor_img, tipo):  # tipo: "orig" ou "rec"
    mean, std = MEAN_STD[tipo]
    ...  # multiplicar por std, somar mean, por canal

def gerar_atribuicao(model, camada, input_tensor, target_class, metodo="ig"):
    if metodo == "ig":
        attr = IntegratedGradients(model).attribute(input_tensor, target=target_class)
    elif metodo == "gradcam":
        attr = LayerGradCam(model, camada).attribute(input_tensor, target=target_class)
    elif metodo == "occlusion":
        attr = Occlusion(model).attribute(
            input_tensor, target=target_class,
            sliding_window_shapes=(3, 16, 16), strides=(3, 8, 8)
        )
    return attr
```

Rode isso para os 4 branches — cada um é um modelo de entrada única (mobilenet/effnet recebendo original **ou** RP), então o Captum se aplica direto, sem adaptação para "entrada dupla".

**Observação relevante para a interpretação depois (não é bug, é um detalhe do `train.py` que vale documentar no artigo):** o comentário em `train.py` linha 127 diz que para RecPlot "inicialização aleatória (treina do zero) é o ideal", mas o código de fato usa `peso_carregado='DEFAULT'` — ou seja, **os pesos convolucionais iniciais também vêm da ImageNet para o branch RecPlot**, mesmo o RecPlot não sendo uma imagem natural. Isso é importante para os *sanity checks* da Etapa 4: um viés herdado da ImageNet (ex. detectores de borda nas primeiras camadas) pode aparecer no branch RecPlot mesmo sem ter "sentido" para aquele domínio, e o teste de randomização de parâmetros ajuda a checar se isso está de fato influenciando a explicação final ou se as camadas finais destravadas (`trainable_blocks=7`) já dominam a decisão.

---

## Etapa 3 — Mapeamento descritor → região do RecPlot (`model/xai/recplot_mapping.py`)

**Ponto de atenção específico ao seu pipeline:** o `dataset.py` carrega as imagens de RecPlot **já salvas em disco** (via `PIL.Image.open`), e só então `transform_recplot` (`utils.py`) aplica `transforms.Resize((224,224))`. Ou seja, existem potencialmente **dois resizes em sequência**:

1. O resize feito no seu código de geração do RecPlot (fora deste repositório, o que você descreveu como resize via sklearn), que converte a matriz N×N de descritores (ex. 100×100) para o tamanho salvo em disco.
2. O `transforms.Resize((224,224))` do `utils.py`, que converte do tamanho salvo em disco para 224×224 (o tamanho que a rede espera).

Se o tamanho salvo em disco já for 224×224, o passo 2 não faz nada (mas ainda passa pela interpolação bilinear padrão do `torchvision.transforms.Resize`). Se for outro tamanho, os dois resizes se compõem.

**Ação prática antes de codar o mapeamento:** confirme o tamanho em que os arquivos de RecPlot estão salvos em disco (`PIL.Image.open(caminho).size` em um arquivo de exemplo). Com esse número, o mapeamento pixel→descritor é a composição de duas transformações lineares:

```python
# model/xai/recplot_mapping.py
def pixel_to_descriptor(row, col, n_descritores=100, tamanho_salvo=None, tamanho_final=224):
    # se tamanho_salvo é None, assume-se que não há resize intermediário (mesmo tamanho da rede)
    escala_disco = n_descritores / (tamanho_salvo or tamanho_final)
    escala_final = (tamanho_salvo or tamanho_final) / tamanho_final
    fator_total = escala_disco * escala_final  # = n_descritores / tamanho_final quando tamanho_salvo é None
    return int(row * fator_total), int(col * fator_total)
```

Isso alimenta diretamente a seção 4 do plano de XAI (overlay de bandas + validação por oclusão de banda) — a função acima é a peça que faltava para ligar "pixel importante no heatmap" a "descritor de percolação específico".

---

## Etapa 4 — Sanity checks (`model/xai/sanity_checks.py`)

**Vantagem do seu código atual:** `model.py` já separa `criar_modelo` (arquitetura + pesos aleatórios/ImageNet) de `carregar_modelo` (arquitetura + pesos treinados). Isso é exatamente o que o teste de randomização de parâmetros de Adebayo et al. precisa — comparar explicação do modelo treinado com a do modelo "do mesmo jeito, mas sem treino" — e você não precisa escrever nenhuma lógica de criação de modelo nova:

```python
from ..model import criar_modelo, carregar_modelo

modelo_treinado = carregar_modelo(backbone, num_classes, tipo, caminho_pesos)
modelo_aleatorio = criar_modelo(backbone, num_classes, pretrained=None)  # pesos aleatórios, mesma arquitetura

attr_treinado  = gerar_atribuicao(modelo_treinado,  camada, x, target)
attr_aleatorio = gerar_atribuicao(modelo_aleatorio, camada, x, target)
# comparar attr_treinado vs attr_aleatorio (correlação de Spearman ou SSIM); esperado: baixa similaridade
```

Rode isso pelo menos uma vez por combinação (backbone × representação × dataset) antes de investir tempo nas métricas quantitativas da Etapa 5 — se um branch falhar no teste, o método de atribuição usado ali precisa ser trocado (ex. de Grad-CAM para Integrated Gradients) antes de tirar qualquer conclusão sobre esse branch específico.

---

## Etapa 5 — Métricas quantitativas em lote (`model/xai/xai_metrics.py`)

**Padrão a seguir:** o mesmo formato de `metrics.py` (`mostrar_metricas` + `metrics_to_csv`), para manter consistência com `resultados_testes.csv` / `resultados_testes_mean.csv` que você já tem. Reaproveite a constante `SEEDS` de `utils.py` — não redeclare a lista de seeds em outro lugar.

```python
# model/xai/xai_metrics.py
from ..utils import SEEDS, DEVICE
import quantus

def xai_metrics_to_csv(seeds, branches_por_seed, test_loader, output_dir):
    linhas = []
    for seed in seeds:
        for nome_branch, model in branches_por_seed[seed].items():
            # quantus.Faithfulness/.../Robustness recebem model, x, y, a (mapa de atribuição)
            ...
            linhas.append({"seed": seed, "branch": nome_branch, "metrica": "faithfulness", "valor": ...})
    df = pd.DataFrame(linhas)
    df.to_csv(f"{output_dir}/xai_metrics.csv", index=False)

    df_mean = df.groupby(["branch", "metrica"]).agg(media=("valor","mean"), std=("valor","std")).reset_index()
    df_mean.to_csv(f"{output_dir}/xai_metrics_mean.csv", index=False)
```

Saída esperada em `resultados_displasia/xai_metrics.csv` e `resultados_pulmao/xai_metrics.csv` — mesma pasta, mesmo espírito dos CSVs que você já gera, só que descrevendo a qualidade da explicação em vez da qualidade da classificação.

**Aviso de custo computacional:** isso roda para 8 seeds × 4 branches × N amostras de teste × múltiplas métricas do Quantus — é a etapa mais pesada do plano. Rode primeiro em um subconjunto pequeno do conjunto de teste para validar que o pipeline está correto antes de rodar completo.

---

## Etapa 6 — Estudos de caso a partir do `predicoes.csv` que você já tem (`model/xai/case_selection.py`)

**Achado importante ao ler `metrics.py`:** a lista `cenarios` (linhas 54–65) inclui **não só as 6 combinações de ensemble, mas também 4 pares "auto"** — `("MobileNet_Original","MobileNet_Original")`, `("MobileNet_RecPlot","MobileNet_RecPlot")`, etc. Matematicamente, somar a probabilidade de um modelo consigo mesmo e renormalizar devolve exatamente a probabilidade original — ou seja, **essas 4 linhas do `predicoes.csv` já são a performance de cada modelo individual**, sem ensemble nenhum. Isso significa que você **não precisa rodar inferência de novo** para comparar "modelo sozinho" vs. "modelo no ensemble": já está tudo no CSV, só filtrar.

```python
import pandas as pd

def casos_corrigidos_pelo_ensemble(predicoes_csv, seed, cenario_individual, cenario_ensemble):
    df = pd.read_csv(predicoes_csv)
    df_seed = df[df.seed == seed]

    individual = df_seed[df_seed.cenario == cenario_individual].set_index("sample")
    ensemble   = df_seed[df_seed.cenario == cenario_ensemble].set_index("sample")

    erro_sozinho = individual.y_pred != individual.y_true
    acerto_ensemble = ensemble.y_pred == ensemble.y_true

    casos = individual.index[erro_sozinho & acerto_ensemble]
    return casos.tolist()

# ex.: onde MobileNet_RecPlot sozinho erra mas o ensemble MobileNet_Original + MobileNet_RecPlot acerta
casos = casos_corrigidos_pelo_ensemble(
    "resultados_displasia/predicoes.csv", seed=7,
    cenario_individual="MobileNet_RecPlot + MobileNet_RecPlot",
    cenario_ensemble="MobileNet_Original + MobileNet_RecPlot",
)
```

O índice `sample` retornado é a posição em `test_dataset.data` (a lista interna de `EnsembleTestDataset`) — depois da correção da Etapa 0 (ordenar os arquivos), `test_dataset.data[sample]["path_orig"]` e `["path_rec"]` te dão os caminhos exatos das imagens daquele caso, prontos para gerar a figura comparativa da seção 7 do plano de XAI.

**Dica de priorização usando os números que você já tem:** olhando `resultados_displasia/resultados_testes_mean.csv`, no dataset de displasia o ensemble `MobileNet_Original + MobileNet_RecPlot` já está em ~99,7% de acurácia média, enquanto `EffNet_RecPlot` sozinho (par "auto") fica em ~84%. Isso sugere focar os primeiros estudos de caso qualitativos em `EffNet_RecPlot`, que é o branch com mais espaço para "aprender algo" nos casos de erro — e comparar com o que `MobileNet_RecPlot` (bem melhor sozinho, ~90%) vê de diferente nas mesmas amostras.

---

## Etapa 7 — Comparando explicações entre arquiteturas (aproveitando a mesma matriz do `metrics.py`)

Com `attributions.py` (Etapa 2) gerando mapas para os 4 branches e `case_selection.py` (Etapa 6) escolhendo amostras, a comparação MobileNet vs. EfficientNet (RQ2 do plano de XAI) fica direta: para a mesma amostra e a mesma representação (original **ou** RecPlot), gere o mapa de atribuição de `mobilenet_orig`/`mobilenet_recplot` e de `effnet_orig`/`effnet_recplot`, normalize ambos os mapas (min-max) e calcule IoU do top-10% ou SSIM entre eles.

Cruzar esse número de concordância com a coluna `y_pred` das 6 linhas de ensemble real do `predicoes.csv` (não as "auto") responde diretamente: quando os dois discordam espacialmente, o ensemble tende a acertar mais?

---

## Etapa 8 — Agregação entre as 8 seeds

Sempre que for gerar uma figura ou métrica "por seed", itere sobre `utils.SEEDS` (a constante já existe, é a mesma lista que aparece nos nomes das pastas `results_kfold/<seed>/`) — não hardcode a lista de novo em nenhum módulo novo. Para as figuras qualitativas (mapa médio de atribuição entre as 8 seeds), carregue os 8 modelos do mesmo branch via `loader.py` (Etapa 1) para a mesma amostra e empilhe os mapas antes de tirar média/desvio — ver seção 5 do plano de XAI para o racional completo de por que agregar em vez de escolher "a melhor seed".

---

## Etapa 9 — Rodando nos dois datasets e comparando (RQ3)

Uma assimetria real entre os dois datasets que vale documentar no artigo: `train.py` usa `StratifiedGroupKFold` agrupado por paciente (extraído do nome do arquivo, linhas 116–123) para o dataset de **displasia**, mas o código comentado (linhas 140–144) indica que o dataset de **pulmão** usa `StratifiedKFold` simples, sem agrupamento por paciente. Isso é uma diferença metodológica real entre os dois datasets (não um erro a corrigir agora, mas algo a declarar na seção de limitações do artigo, porque pode inflar levemente a performance no pulmão se houver múltiplas imagens do mesmo paciente espalhadas entre treino e validação).

Para o restante, o pipeline de XAI é genérico o suficiente para não precisar de nenhuma adaptação por dataset: `metrics.py` já lida com número de classes de forma genérica (`for classe in range(probs.shape[1])`), confirmado pelos CSVs — displasia tem `prob_0,prob_1` e pulmão tem `prob_0,prob_1,prob_2`. Os módulos novos (`xai_metrics.py`, `case_selection.py`) seguem o mesmo princípio: parametrizar por `output_dir` (`resultados_displasia` ou `resultados_pulmao`) e `class_names`, sem lógica condicional por dataset.

---

## Ordem de execução recomendada

1. Etapa 0 (ajustes de 10 minutos, mas evita retrabalho).
2. Etapa 1 + 2 (infra + atribuição) num notebook exploratório, só para 1 seed e poucas amostras, para validar visualmente antes de automatizar.
3. Etapa 4 (sanity checks) — barato e pode mudar sua escolha de método antes de investir na Etapa 5.
4. Etapa 3 (mapeamento) em paralelo — não depende das anteriores, só precisa do tamanho salvo em disco do RecPlot.
5. Etapa 5 (métricas em lote) — a mais cara, deixe para quando 1–4 já estiverem validadas.
6. Etapas 6–7 (estudos de caso e comparação de arquiteturas) — usam o que já foi gerado, é majoritariamente análise/figuras.
7. Etapa 8–9 — consolidação para os dois datasets e escrita.

Isso mantém coerência com o cronograma da seção 11 do `plano_xai_recplot_percolacao.md`, agora ajustado ao que já existe no repositório em vez de partir do zero.
