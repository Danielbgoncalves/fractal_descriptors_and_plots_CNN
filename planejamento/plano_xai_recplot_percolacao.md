# Plano de Desenvolvimento: XAI para Descritores Fractais e RecPlots de Percolação

**Objetivo do documento:** transformar o pipeline atual (descritores fractais → RecPlot → MobileNet/EfficientNet → ensemble) em um artigo de computação aplicada a imagens com um eixo central de interpretabilidade (XAI), que permita entender *onde* e *por quê* cada modelo decide, comparando arquiteturas, datasets e representações (original vs. RecPlot).

---

## 1. Perguntas de pesquisa (o esqueleto do artigo)

Antes de codar, vale fixar 4–5 perguntas que os experimentos de XAI vão responder. Isso evita "gerar mapas de saliência por gerar" e dá ao artigo uma narrativa clara. Sugestão de RQs:

1. **RQ1 (o que o RecPlot ensina):** as regiões mais importantes do RecPlot de percolação correspondem a faixas específicas de descritores (ex.: limiares de percolação altos vs. baixos)? Isso é consistente entre classes e datasets?
2. **RQ2 (arquitetura):** MobileNet e EfficientNet "olham" para as mesmas regiões (no original e no RecPlot) quando concordam? E quando discordam?
3. **RQ3 (dataset):** o padrão de importância no RecPlot muda entre displasia (2 classes) e pulmão (3 classes), ou é um padrão genérico do algoritmo de percolação?
4. **RQ4 (original vs. RecPlot):** as duas representações trazem evidência complementar (o ensemble ganha porque cada um vê algo diferente) ou redundante?
5. **RQ5 (estabilidade):** o padrão de importância é estável entre as 8 seeds, ou é um artefato de inicialização?

Cada seção abaixo alimenta uma ou mais dessas RQs — vou marcar entre colchetes.

---

## 2. Métodos de atribuição a usar (além do Captum "cru") `[RQ1, RQ2, RQ4]`

Você já usa Captum — ótimo, é a lib certa, só falta variar o método e ser sistemático. Recomendo rodar **mais de um método** por imagem, porque métodos diferentes respondem perguntas diferentes e um único método pode enganar (ver seção 3 sobre sanity checks):

- **Integrated Gradients** (`captum.attr.IntegratedGradients`): atribuição pixel a pixel, boa resolução fina — útil para o RecPlot, onde cada pixel tem significado (par de descritores).
- **Grad-CAM / Grad-CAM++** (via `captum.attr.LayerGradCam` ou a lib `pytorch-grad-cam`): localização em nível de região/bloco, mais robusto a ruído pixel-a-pixel, bom para a imagem histológica original.
- **Occlusion** (`captum.attr.Occlusion`): sensibilidade a blocos — no RecPlot, ocluir blocos que correspondem a **faixas inteiras de descritores** (não só blocos genéricos quadrados) é uma validação direta e barata da RQ1 (ver seção 4).
- **SHAP (GradientExplainer/DeepExplainer)**: opcional, mais custoso, mas dá uma segunda base teórica (valores de Shapley) para triangular com Integrated Gradients.

Ideia prática: não escolha "o melhor método" a priori — reporte 2 (IG + Grad-CAM) como principais e use os outros como checagem de robustez de conclusão.

**Cuidado com camadas equivalentes:** ao comparar MobileNet vs. EfficientNet com Grad-CAM, escolha a **última camada convolucional de cada rede** de forma consistente (ex. último bloco antes do global pooling em ambas), documente explicitamente qual camada foi usada em cada arquitetura — isso é frequentemente esquecido e vira crítica de revisor.

---

## 3. Avaliação quantitativa das explicações — não só "olhar o heatmap" `[RQ2, RQ4, RQ5]`

Isso é o que vai dar peso científico ao artigo (a maioria dos trabalhos aplicados só mostra imagens bonitas de Grad-CAM; um revisor de peso vai cobrar métrica).

**Biblioteca recomendada:** [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus), que <cite index="7-1">é um toolkit com mais de 30 métricas organizadas em categorias como fidelidade (faithfulness), robustez, localização, complexidade e randomização</cite>. Ela é agnóstica ao método de atribuição, então dá pra plugar IG, Grad-CAM e Occlusion e comparar todos com os mesmos números. Métricas centrais a reportar:

- **Faithfulness** (ex. Pixel-Flipping / Faithfulness Correlation): mascara pixels em ordem de importância e mede a queda no logit da classe — se a queda é rápida, a explicação é fiel ao que o modelo realmente usa.
- **Robustness**: pequenas perturbações no input geram explicações parecidas? (isso conversa com a comparação Original vs. RecPlot — RecPlot pode ser mais sensível a ruído).
- **Localization**: aqui você tem uma vantagem rara — pode construir um "gabarito" a partir do seu conhecimento do algoritmo de geração do RP (seção 4), algo que a maioria dos papers de XAI em imagens médicas não tem (não há máscara de especialista).
- **Complexity/Sparseness**: a explicação está concentrada em poucas regiões (interpretável) ou espalhada (difícil de interpretar)?

**Sanity checks (não pule isso):** rode o teste de randomização de parâmetros de Adebayo et al. — <cite index="14-1">o teste compara a saliência de um modelo treinado com a saliência do mesmo modelo com pesos re-inicializados aleatoriamente</cite>, camada por camada. Se o mapa de Grad-CAM continua "bonito" mesmo com pesos aleatórios, o método está te enganando e isso precisa ser reportado (ou trocado de método) antes de tirar qualquer conclusão biológica. É um teste barato (algumas horas de execução) que blinda o artigo de uma crítica clássica de revisor de XAI.

---

## 4. Mapeamento descritor → região do RecPlot (o ponto mais valioso que você já tem) `[RQ1 — provavelmente a maior contribuição do artigo]`

Você mencionou que sabe, pelo algoritmo, a que faixa de descritores cada região do RP corresponde (mesmo sem precisão de pixel). Isso é ouro para XAI e pouquíssimos trabalhos de RP+CNN exploram isso — vale transformar em método formal, não em observação informal do texto.

**O que construir:**

1. Uma função `pixel_to_descriptor(i, j, n_descritores=100, tamanho_resize=224)` que, dado o fator de resize (sklearn `resize`), devolve o par de índices de descritor original `(i_orig, j_orig)` correspondente a um pixel `(i, j)` do RP redimensionado. Como o resize é uma interpolação determinística, o mapeamento inverso é simples (`i_orig = i * n_descritores / tamanho_resize`).
2. Uma tabela de "bandas" (`bandas_descritores.csv`): quais índices de descritor são percolação/lacunaridade/dimensão fractal, e dentro de percolação, quais correspondem a limiares baixos/médios/altos. Isso vira uma legenda fixa sobreposta ao heatmap.
3. Uma função de overlay: heatmap de atribuição + grade com os rótulos de banda, para as figuras qualitativas do artigo (muito mais informativo que "região quente no canto superior esquerdo").

**Validação quantitativa dessa correspondência (não fique só no qualitativo):**

- **Occlusion por banda:** ocluir (substituir pela média ou por ruído) blocos inteiros do RP correspondentes a uma banda de descritores e medir a queda de acurácia/logit — se a banda de percolação alta cai mais que a de baixa, você tem evidência quantitativa, não só visual, de que "o modelo aprende com o limiar de percolação alto".
- **Correlação atribuição↔descritor:** para cada amostra, somar a "massa" de atribuição (IG ou Grad-CAM) dentro de cada banda e correlacionar (Spearman) com o valor bruto do descritor correspondente, através do dataset. Se atribuição alta em uma banda anda junto com valores extremos daquele descritor, isso é uma frase forte para a seção de resultados.

Essa seção sozinha já daria para uma subseção de resultados robusta e é o diferencial do seu trabalho frente a "mais um paper de Grad-CAM em histologia".

---

## 5. Como lidar com as 8 seeds (não escolha "a melhor") `[RQ5]`

Sua intuição de desconfiar está certa: escolher a seed com melhor F1 para gerar as figuras de XAI é *cherry-picking* e um revisor vai questionar. Só a seed muda entre as 8 execuções, então trate-as como uma amostra de variabilidade, não como 8 modelos a serem ranqueados.

**Proposta de dois níveis:**

- **Quantitativo (todas as métricas de XAI da seção 3):** rode para as 8 seeds e reporte média ± desvio padrão, exatamente como já faz para F1/acc no `resultados_testes_mean.csv`. Crie um `xai_metrics.csv` seguindo o mesmo padrão. Isso já responde RQ5 de forma direta: desvio padrão baixo entre seeds = explicação estável = conclusão mais forte.
- **Qualitativo (as figuras "bonitas" de heatmap no artigo):** não dá pra colocar 8 imagens por combinação. Duas opções, ambas defensáveis:
  1. **Mapa agregado:** média (e desvio-padrão como mapa de "incerteza da explicação") dos heatmaps das 8 seeds para a mesma amostra — trata a explicação como um ensemble também, coerente com o resto do seu pipeline.
  2. **Seed representativa:** a seed cujo F1 é o **mais próximo da média das 8** (não a melhor) — documente esse critério explicitamente no artigo ("seed mediana por F1", não "melhor seed").

Recomendo usar (1) para as figuras principais e (2) como fallback quando o mapa agregado ficar visualmente poluído (ex. Occlusion, que é mais binário).

**Bônus de RQ5:** o desvio-padrão entre seeds é, na prática, um teste de robustez mais fraco que o "sanity check" da seção 3, mas é gratuito (você já tem os 8 modelos treinados) — vale reportar como evidência extra de estabilidade.

---

## 6. Matriz de comparações cross-cutting `[RQ2, RQ3, RQ4]`

Para não se perder, formalize a matriz de experimentos de XAI como uma tabela (linhas = o que muda, colunas = o que é medido). Sugestão de fatores a cruzar:

| Fator | Níveis | Pergunta que responde |
|---|---|---|
| Arquitetura | MobileNet, EfficientNet | RQ2 |
| Representação | Original, RecPlot | RQ4 |
| Dataset | Displasia (2 classes), Pulmão (3 classes) | RQ3 |
| Classe (dentro de cada dataset) | ex. saudável vs. severo | RQ1 (padrão muda por classe?) |
| Seed | 1–8 (agregado, ver seção 5) | RQ5 |

Cada célula relevante dessa matriz gera uma linha do `xai_metrics.csv` (faithfulness, robustness, localization, complexity) + um conjunto de mapas agregados. Isso vira naturalmente as tabelas e figuras da seção de Resultados.

**Métrica de concordância entre arquiteturas (RQ2):** para cada amostra, compare o heatmap de MobileNet com o de EfficientNet via IoU do top-k% mais importante (ex. top 10%) ou SSIM entre os mapas normalizados. Depois cruze isso com o CSV de predições: quando os dois discordam espacialmente, o ensemble tende a acertar mais? Isso conecta diretamente XAI com a justificativa estatística do ensemble — provavelmente o argumento mais forte do artigo todo.

---

## 7. Explicando o ensemble, não só os modelos isolados `[RQ4]`

Como o ensemble parece ser um soft-voting (média das probabilidades dos dois ramos, a julgar pelo formato do `predicoes.csv`), a explicação de cada ramo é simplesmente a explicação do modelo individual — não precisa de método novo. O que precisa de desenho de experimento é a **narrativa de complementaridade**:

1. Usando o `predicoes.csv`, identifique amostras onde **um modelo erra sozinho mas o ensemble acerta** (o outro ramo "corrigiu"). Você já tem os logits por configuração, então isso é só um filtro em pandas.
2. Para essas amostras, gere a explicação de cada ramo lado a lado (original com Grad-CAM/IG do MobileNet-Original, RecPlot com IG do MobileNet-RecPlot, etc.) e mostre visualmente *o que o ramo "salvador" viu que o outro não viu*.
3. Selecione 2–4 casos assim por dataset para virarem figuras de estudo de caso no artigo — é o tipo de figura que mais convence revisor de aplicação clínica, mais que qualquer heatmap solto.

---

## 8. Estudos de caso a partir do que você já tem (`predicoes.csv`) `[transversal]`

Como você já registra logits por amostra e por configuração, dá pra selecionar casos automaticamente em vez de escolher à mão (evita viés de seleção). Sugestão de categorias a extrair programaticamente:

- Acerto de alta confiança (prob > 0.9) — "caso típico", bom para mostrar o padrão esperado.
- Erro de alta confiança — o modelo está "certo de algo errado"; ver o que a explicação mostra ali é frequentemente a parte mais interessante do artigo.
- Casos de fronteira (prob próxima de 0.5) — mostra o que confunde o modelo.
- Casos de discordância entre ramos (seção 7).

Automatizar essa seleção (uma função `selecionar_casos(predicoes_df, criterio)`) também facilita repetir o processo para os dois datasets sem retrabalho manual.

---

## 9. Extensão sugerida do código

Dado o que já existe (`dataset.py`, `metrics.py`, `model.py`, `run.py`, `train.py`, `utils.py`), sugiro isolar o XAI em módulos próprios, sem mexer no pipeline de treino:

```
model/
├── xai/
│   ├── attributions.py      # wrappers de Captum: IG, GradCAM, Occlusion por modelo/branch
│   ├── recplot_mapping.py   # pixel_to_descriptor(), bandas_descritores, overlay de legenda
│   ├── xai_metrics.py       # wrappers de Quantus (faithfulness/robustness/localization) + sanity checks
│   ├── seed_aggregation.py  # agregação de heatmaps entre as 8 seeds (média/desvio)
│   └── case_selection.py    # filtros sobre predicoes.csv (acerto/erro/discordância)
```

Saídas esperadas, seguindo o padrão que você já usa para métricas de classificação:
- `xai_metrics.csv` (uma linha por: modelo, representação, dataset, seed, métrica, valor) — igual em espírito ao `resultados_testes.csv`.
- `xai_metrics_mean.csv` (agregado por seed, igual ao `resultados_testes_mean.csv`).
- Pasta de figuras `xai_figuras/` com os overlays anotados por banda de descritor.

Isso mantém a mesma filosofia de "tudo em CSV, quase tudo pode ser recalculado depois" que você já tem.

---

## 10. Estrutura sugerida do artigo

1. **Introdução** — motivação: descritores fractais/percolação + ensemble multi-arquitetura têm bom desempenho, mas faltam trabalhos que expliquem *por que* funcionam, especialmente para a representação RecPlot, que não é uma imagem "natural" e por isso precisa de interpretação própria.
2. **Trabalhos relacionados** — descritores fractais/percolação em imagens histológicas; recurrence plots como representação para CNN; XAI em imagens médicas (Grad-CAM, IG); avaliação quantitativa de XAI (Quantus, sanity checks de Adebayo).
3. **Materiais e Métodos** — pipeline de extração de descritores → RP (breve, referenciando trabalho anterior se já publicado); arquiteturas e ensemble; datasets; **métodos de atribuição usados**; **métricas de avaliação de XAI**; **protocolo de agregação das 8 seeds** (documentar o critério da seção 5 explicitamente, é comum revisor perguntar).
4. **Resultados** — 4.1 métricas quantitativas de XAI por célula da matriz (seção 6); 4.2 mapeamento descritor↔região e validação por oclusão de banda (seção 4); 4.3 concordância MobileNet vs. EfficientNet e relação com o ensemble (seção 6/7); 4.4 estudos de caso (seção 7/8); 4.5 sanity checks e estabilidade entre seeds.
5. **Discussão** — o que os padrões de percolação "significam" clinicamente (com cautela — você não é patologista, então enquadre como hipótese a validar, não afirmação diagnóstica); diferenças entre datasets; limitações.
6. **Conclusão e trabalhos futuros** — ex. validação com anotação de especialista, extensão para lacunaridade/dimensão fractal como próprias representações RP.

---

## 11. Cronograma sugerido (ordem de prioridade, não datas fixas)

1. `recplot_mapping.py` + validação da correspondência pixel↔descritor (base de tudo, sem isso a seção 4 não existe).
2. `attributions.py` com IG + Grad-CAM funcionando para os 4 branches (Original/RP × MobileNet/EffNet), qualitativo primeiro.
3. Sanity checks de Adebayo (barato, roda em horas, e pode invalidar/confirmar a escolha de método antes de investir tempo em mais análises).
4. `xai_metrics.py` (Quantus) rodando em lote para as 8 seeds × 4 branches × 2 datasets — a parte computacionalmente mais pesada, planeje tempo de GPU para isso.
5. Oclusão por banda + correlação atribuição↔descritor (seção 4, parte quantitativa).
6. Concordância entre arquiteturas + estudos de caso do ensemble (seções 6/7/8).
7. Consolidar tabelas/figuras finais e escrever.

---

## 12. Cuidados estatísticos e limitações a declarar

- **Múltiplas comparações:** a matriz da seção 6 gera muitos testes de hipótese (arquitetura × representação × dataset × classe). Use correção (Holm-Bonferroni) se for reportar p-valores, ou prefira reportar tamanhos de efeito (diferença de médias com IC) em vez de só "significativo/não".
- **N pequeno nas seeds:** com 8 seeds, testes não-paramétricos pareados (Wilcoxon signed-rank) são mais apropriados que t-test para comparar, por exemplo, faithfulness de MobileNet vs. EfficientNet entre seeds.
- **Não superinterprete o RecPlot como "verdade biológica":** a correspondência descritor↔pixel é sólida matematicamente (você controla o algoritmo), mas a correspondência descritor↔biologia (o que aquele valor de percolação *significa* no tecido) é uma hipótese, não um fato estabelecido — trate como tal na discussão.
- **Sanity checks primeiro, conclusões depois:** se Grad-CAM falhar no teste de randomização de parâmetros em algum branch, troque a narrativa para IG/Occlusion nesse branch específico, e reporte a falha — isso é visto como rigor, não como fraqueza, por revisores de XAI.

---

### Bibliotecas/ferramentas mencionadas
- `captum` (já em uso) — IG, LayerGradCam, Occlusion, GradientShap.
- `quantus` — avaliação quantitativa de explicações (faithfulness, robustness, localization, complexity).
- `pytorch-grad-cam` (jacobgil) — alternativa/complemento ao Captum para variantes de CAM (Grad-CAM++, Score-CAM).
- `scipy.stats` — Wilcoxon signed-rank, correlação de Spearman.
