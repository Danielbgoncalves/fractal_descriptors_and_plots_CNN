# RecPlot de Percolação em imagens de Displasia Oral e Câncer de Pulmão

---

## Objetivo

Avaliar o impacto do uso da representação por **Recurrence Plot (RP)** na classificação de imagens histológicas de displasia oral.

Foram comparados:

- Imagens originais;
- Imagens geradas por Recurrence Plot;
- Diferentes arquiteturas de CNN;
- Combinações em ensemble.

---

## Configuração Experimental

Foram avaliados:

- MobileNet_v2
- EfficientNet_b0

Representações:

- Imagem Original
- Recurrence Plot (RP)

Cada configuração foi executada utilizando 8 *seeds*, permitindo analisar não apenas a média das métricas, mas também sua estabilidade.

---

## Exemplos de Imagens

### Displasia
<p align="center">
  <img src="imgs\healthy-01-roi1.png" width="45%">
  <img src="../saida_new\RP\treino_validacao\healthy\healthy-01-roi1.png" width="25%">
</p>

### Pulmão
<p align="center">
  <img src="imgs/aca_md_20x_0.jpg" width="39%">
  <img src="../RP_perc_pulmao/RP/teste/aca_md/aca_md_20x_0.png" width="29%">
</p>

---

## Observações sobre o treinamento

* As redes apresentaram forte tendência ao overfitting durante o treinamento.
* A aplicação de data augmentation (apagamento aleatório de regiões e pequenas variações de brilho) piorou a qualidade das predições.
* O congelamento das camadas iniciais da EfficientNet e da MobileNet reduziu o overfitting e melhorou o desempenho dos modelos.
* A EfficientNet mostrou-se mais propensa ao overfitting do que a MobileNet, provavelmente devido à sua maior complexidade, o que dificulta o treinamento com um conjunto reduzido de imagens.

---

## Graficos! 
<h2 align="center">acc vs f1</h2>

<div style="display:flex; justify-content:center; gap:20px;">
  <div align="center">
    <b>Displasia</b><br>
    <img src="..\resultados\teste_padrao\plots_Displasia_2_classes\acc_vs_f1.png" width="500">
  </div>

  <div align="center">
    <b>Pulmão</b><br>
    <img src="..\resultados\teste_padrao\plots_Lung_3_classes\acc_vs_f1.png" width="500">
  </div>
</div>

---

<h2 align="center">Boxplots</h2>
<p align="center">
<b>Acurácia média e desvio padrão para Displasia.</b><br>
<img src="..\resultados\teste_padrao\plots_Displasia_2_classes\boxplot_acc.png" width="700">
</p>

<p align="center">
<b>F1 Macro médio e desvio padrão para Pulmão.</b><br>
<img src="..\resultados\teste_padrao\plots_Lung_3_classes\boxplot_f1_macro.png" width="700">
</p>

---

## XAI
Para compreender quais regiões influenciaram as decisões das redes, foi utilizada a biblioteca **Captum**.

### Visão Geral

#### Displasia MobileNet
<div style="display:flex; justify-content:center; gap:20px;">
  <div align="center">
    <b>Saudável</b><br>
    <img src="imgs\heatmap_healthy_mob.png" width="500" height="420">
  </div>

  <div align="center">
    <b>Severe</b><br>
    <img src="imgs\heatmap_severe_mob.png" width="500" height="420">
  </div>
</div>

#### Displasia EfficientNet
<div style="display:flex; justify-content:center; gap:20px;">
  <div align="center">
    <b>Saudável</b><br>
    <img src="imgs\heatmap_healthy_eff.png" width="500" height="420">
  </div>

  <div align="center">
    <b>Severe</b><br>
    <img src="imgs\heatmap_severe_eff.png" width="500" height="420">
  </div>
</div>

#### Pulmão MobileNet
<div style="display:flex; justify-content:center; gap:20px;">
  <div align="center">
    <b>Aca</b><br>
    <img src="imgs\heatmap_aca_mob.png" width="400" height="320">
  </div>

  <div align="center">
    <b>Nor</b><br>
    <img src="imgs\heatmap_nor_mob.png" width="400" height="320">
  </div>

  <div align="center">
    <b>Scc</b><br>
    <img src="imgs\heatmap_scc_mob.png" width="400" height="320">
  </div>
</div>

#### Pulmão EfficientNet
<div style="display:flex; justify-content:center; gap:20px;">
  <div align="center">
    <b>Aca</b><br>
    <img src="imgs\heatmap_aca_eff.png" width="400" height="320">
  </div>

  <div align="center">
    <b>Nor</b><br>
    <img src="imgs\heatmap_nor_eff.png" width="400" height="320">
  </div>

  <div align="center">
    <b>Scc</b><br>
    <img src="imgs\heatmap_scc_eff.png" width="400" height="320">
  </div>
</div>

<br>

> Tradução:

![relação descritores por posição](imgs/mapa_descritores.png)

$P_k$ — Número médio de clusters:
Mede a quantidade média de agrupamentos (componentes conexos) formados em cada janela da imagem. Valores altos indicam uma textura mais fragmentada, enquanto valores baixos sugerem regiões mais contínuas e homogêneas.
$G_k$— Probabilidade de percolação:
Mede a fração de janelas que atingem o limiar crítico de percolação, indicando a formação de uma estrutura suficientemente conectada. Valores altos representam maior conectividade da textura, enquanto valores baixos indicam predominância de regiões fragmentadas.
$H_k$ — Tamanho médio do maior cluster:
Mede a proporção ocupada pelo maior agrupamento em cada janela. Valores elevados indicam a presença de grandes regiões conectadas, enquanto valores baixos refletem agrupamentos menores e mais dispersos.

### Visão em imagens específicas

#### Displasia: MobileNet × EfficientNet

<div align="center">

<div style="display: flex; justify-content: center; gap: 40px;">

<div align="center">
<b>MobileNet</b><br><br>
<b>Classe Saudável</b><br>
<img src="imgs/healthy_10_roi2_mob.png" width="550">
</div>

<div align="center">
<b>EfficientNet</b><br><br>
<b>Classe Saudável</b><br>
<img src="imgs/healthy-11-roi3_eff.png" width="550">
</div>

</div>

<br>

<div style="display: flex; justify-content: center; gap: 40px;">

<div align="center">
<b>Classe Severe</b><br>
<img src="imgs/severe_08_roi5_mob.png" width="550">
</div>

<div align="center">
<b>Classe Severe</b><br>
<img src="imgs/severe-05-roi4_eff.png" width="550">
</div>

</div>

</div>

#### Pulmão: MobileNet × EfficientNet

<div align="center">

<div style="display: flex; justify-content: center; gap: 40px;">

<div align="center">
<b>MobileNet</b><br><br>
<b>Classe Acc</b><br>
<img src="imgs/aca_md_20x_73_mob.png" width="550">
</div>

<div align="center">
<b>EfficientNet</b><br><br>
<b>Classe Acc</b><br>
<img src="imgs/aca_md_20x_73.png" width="550">
</div>

</div>

<br>

<div style="display: flex; justify-content: center; gap: 40px;">

<div align="center">
<b>Classe Nor</b><br>
<img src="imgs/nor_40x_28_mob.png" width="550">
</div>

<div align="center">
<b>Classe Nor</b><br>
<img src="imgs/nor_40x_28.png" width="550">
</div>

</div>

<br>

<div style="display: flex; justify-content: center; gap: 40px;">

<div align="center">
<b>Classe Scc</b><br>
<img src="imgs/scc_md_20x_24_mob.png" width="550">
</div>

<div align="center">
<b>Classe Scc</b><br>
<img src="imgs/scc_md_20x_24.png" width="550">
</div>

</div>

</div>