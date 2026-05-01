# Benchmark 

Esse diretório foi usado para medir, antes de tudos os valores de retorno das principais funções do projeto, e depois de cada otimização, para verificar se houve algum ganho de desempenho.

Usou-se a biblioteca `time` do Python para medir o tempo de execução de cada função, Numba foi usada para "compilar" alguams funções e para paralelizar algumas operações.

## Resultados


### 📊 Resultados da Otimização

| Função           | Tempo Original (s) | Tempo Otimizado (s) | Speedup |
|------------------|--------------------|---------------------|---------|
| Mink_perc        | 5.17               | 1.05                | 4.88x   |
| Eucl_perc        | 4.63               | 0.97                | 4.76x   |
| Manh_perc        | 3.83               | 0.87                | 4.38x   |
| MatrizProb       | 0.88               | 0.22                | 4.01x   |
| MatrizProbEucl   | 0.63               | 0.15                | 4.18x   |
| MatrizProbManh   | 0.53               | 0.15                | 3.64x   |
| LAC              | 0.0003             | 0.0002              | 1.22x   |

---

### 📌 Totais por Imagem
- **Tempo Original:** 15.7 s  
- **Tempo Otimizado:** 3.43 s  
- **Speedup Médio:** 4.5x  

---

### ✅ Observações
- As funções **Mink_perc**, **Eucl_perc** e **Manh_perc** tiveram os maiores ganhos de desempenho (speedup > 4x).  
- A função **LAC** já era extremamente rápida, por isso o ganho foi marginal.  
- A otimização reduziu o tempo total por imagem em **~78%**, mostrando grande impacto na eficiência geral.

