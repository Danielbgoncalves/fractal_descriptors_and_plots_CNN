# Validação de equivalência entre plots (Python vs MATLAB)

Este documento resume a validação automática realizada comparando imagens geradas pelo código Python e pelo código MATLAB, baseada no arquivo `resultado_comparacao.csv` gerado pelo script de comparação.

**Objetivo:**
- Verificar se as imagens geradas pelas implementações Python e MATLAB são visualmente equivalentes, usando o índice SSIM (Structural Similarity).

**Método:**
- Foram utilizadas as 50 primeiras imagens do dataset de Displasias em sua subpasta Healthy.
- Leitura das subpastas comuns entre as pastas de saída do Python e do MATLAB.
- Para cada arquivo com nome correspondente em ambas as pastas, calcular o SSIM entre as imagens (após leitura com OpenCV e conversão para RGB).
- Normalização das imagens antes da comparação.
- Classificação como `equivalente` quando `ssim >= 0.95`, caso contrário `diferente`. Arquivos com dimensões diferentes são marcados como `dimensoes_diferentes` e ignorados na comparação de SSIM.

**Paths usados (script):**
(criadas com o intuíto deste teste)
- Saída Python: `codigo-fractal-python/feature_to_image/saida3`
- Saída MATLAB: `codigos-fractal-matlab/saida3`

**Resumo dos resultados (baseado em `resultado_comparacao.csv`):**
- Total de comparações: 100
- `equivalente`: 100
- `diferente`: 0
- `dimensoes_diferentes`: 0

- Contagem de SSIM computados: 100
- SSIM médio: 0.999254949040
- SSIM mínimo: 0.998110865669 — arquivo: `F-RecPlot/F-RecPlot50.png`
- SSIM máximo: 0.999929137291 — arquivo: `F-Classical/F-Classical42.png`

**Interpretação:**
- Todas as 100 imagens comparadas foram classificadas como `equivalente` com o limiar de 0.95, e os valores de SSIM são muito elevados (típicos entre ~0.9981 e ~0.9999), indicando alta similaridade visual entre as saídas Python e MATLAB.

**Observações e próximos passos recomendados:**
- Mesmo com SSIM alto, revisar visualmente amostras (especialmente os extremos — menor e maior SSIM) pode ajudar a identificar pequenas diferenças que o índice não penaliza.
- Se for necessário detectar diferenças sutis, considerar reduzir o limiar (ex.: 0.99) ou usar métricas adicionais (diferença absoluta por pixel, PSNR, histogramas).
- Garantir que o pré-processamento (normalização, conversão de canais) seja idêntico entre implementações para evitar discrepâncias artificiais.

**Conclusão:**
- A validação atual mostra que as implementações Python e MATLAB produzem resultados equivalentes para o conjunto verificado (100 imagens), com SSIM médio de 0.99925.