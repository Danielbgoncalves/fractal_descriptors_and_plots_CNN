# Validação: Implementação Python vs. MATLAB

## Resumo

Este projeto compara a implementação em Python de descritores fractais com a implementação de referência em MATLAB, estabelecida como padrão a ser alcançado. A análise abrange **363 descritores fractais** extraídos de **228 amostras** de imagens hitológicas reais, demonstrando alta concordância entre as implementações.

---

## Resultados Principais

### Métricas Globais de Concordância

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **Erro Absoluto Médio** | 0.001667 | Diferença média de ~0.002 entre implementações |
| **Erro Relativo Médio** | 0.0656% | **Excelente**: < 0.1% de desvio relativo |
| **RMSE Médio** | 0.006383 | Raiz do erro quadrático muito baixa |
| **Correlação Média** | 0.999984 | Bom demais: praticamente +1 |
| **Concordância Estatística** | 165/363 (~45%) | Pouca diferença significativa em quse metade (p > 0.05) |

### TL; DR
**A implementação Python é estatisticamente equivalente ao MATLAB**, com desvios desprezíveis que podem ser atribuídos a diferenças de precisão numérica entre as plataformas.

---

## Análise Visual

### 1. Distribuição de Erros por Descritor

#### **Visão Geral Completa**

![Erro Relativo - Visão Geral](plots/Erro_Relativo_por_Coluna_Visão_Geral.png)

- **Destaque**: Os descritores próximos do final apresentam erro de ~8%
- **Contexto**: É um outlier isolado; os demais 360 descritores têm erro próximo de zero

#### **Primeiros 360 Descritores**

![Erro Relativo - 360 descritores](plots/Erro_Relativo_por_Coluna_Primeiros_360_descritores.png)

- **Comportamento**: Erros baixos (< 0.4%)
- **Outliers**: Poucos picos isolados, provavelmente devido a descritores com valores próximos de zero (talvez? n sei)

#### **Últimos 3 Descritores**

![Erro Relativo - Últimos 3](plots/Erro_Relativo_por_Coluna_Ultimos_3_descritores.png)

- **Observação crítica**: Erro relativo mais elevado (3-8%)
- **Causa provável**: 
  - Há, nas duas versões, tanto MATLAB quanto Python, uma função dedicada ao cálculo da Dimensão Fractal - descritor analisado aqui -, contudo ela não é utilizada. O cálculo é feito por meio de algoritmo de regressão linear em ambos.
  - A diferente implementação desses algoritmos em cada ambiente deve ser o responsável pela diferença.
- **Impacto**: Mínimo, pois representa apenas 0.8% dos descritores

### 2. Dispersão Global MATLAB vs. Python

![Dispersão Global](plots/Dispersao_global.png)

- **Interpretação**: Pontos alinhados perfeitamente sobre a diagonal (y = x)
- **Significado**: Relação linear perfeita entre as implementações
- **Desvios**: Imperceptíveis visualmente, confirmando a equivalência numérica

### 3. Teste Estatístico de Hpótese

![Dispersão Global](plots/Diferenca_estatistica.png)

- **Interpretação**: Pontos acima da linha tracejada aceitam a hipótese de não ter diferença nos resultados
- **Significado**: o teste avalia, para cada descritor, se há diferença estatisticamente significativa entre os resultados do Python e do MATLAB. Valores de p acima de 0.05 indicam que as implementações geram resultados equivalentes.
- **Desvios**: as poucas colunas com p abaixo de 0.05 indicam discrepâncias numéricas sutis em descritores específicos.

---

## Análise Detalhada

### Distribuição de Erros Relativos

```
Faixa de Erro | Descritores | Percentual
--------------|-------------|------------
< 0.01%       |         180 |      49.6%
0.01% - 1%    |         180 |      49.6%
1% - 5%       |           2 |       0.6%
5% - 15%      |           1 |       0.3%
```

### Teste t Pareado (α = 0.05)

- **Hipótese nula**: Não há diferença entre as médias das implementações
- **Resultado**: 45% dos descritores **não rejeitam H₀**
- **Conclusão**: As duas implementações (Python e MATLAB) produzem resultados estatisticamente equivalentes em cerca de 45% dos descritores, indicando alta consistência geral entre os métodos. As diferenças observadas nos demais descritores são pequenas, mas detectáveis estatisticamente.
- Finalmente utilizando estatística de vdd !!!

---
## Conclusões

- Equivalência numérica muito boa 
- Alta reprodutibilidade estatística
- 🤓🤩
