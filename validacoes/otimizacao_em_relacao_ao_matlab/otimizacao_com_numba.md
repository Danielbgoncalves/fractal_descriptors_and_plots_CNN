## Otimizações de Performance

Durante a validação inicial, a implementação em Python apresentou tempos de execução muito superiores ao MATLAB, especialmente nas funções da família **`pmr`**, que são naturalmente intensivas em operações numéricas.  

Foram investigadas diferentes abordagens de otimização, incluindo **vetorização com NumPy**, **paralelismo**, e finalmente o uso do **Numba**, que se mostrou a alternativa mais eficiente e estável.

### Funções `pmr`

Essas funções estavam entre os maiores gargalos de desempenho. Para imagens de **224×224 pixels** e `maxr = 41`, cada execução levava cerca de **10 minutos**.  

Tentativas de **vetorização com NumPy** resultaram em estouros de memória — algumas tentativas resultaram em operações que exigiam alocação de quase **2 GB** em arrays temporários, tornando a abordagem inviável.  

A solução definitiva foi o uso do **decorador `@njit` do Numba**, que converte o código Python puro em código nativo otimizado (similar a C). Essa simples modificação reduziu o tempo de execução de **minutos para segundos**, mantendo exatamente o mesmo resultado numérico.

### Funções `clustperc`

Nessa parte do código, houve necessidade de substituir a função de rotulagem de componentes conectados da biblioteca `scikit-image` (`label`) por uma implementação customizada, chamada **`_label_4conn`**. Isso ocorre porque numba otimiza o python puro e mais alguams bibliotecas como numpy, métodos da scikit-image impediriam sua aplicação.

Essa nova versão foi escrita em Python puro, mas com o **Numba** aplicado para compilação JIT (Just-In-Time). Além de eliminar dependências externas, ela trouxe grande ganho de velocidade ao evitar overheads de interface entre Python e C durante laços internos.

### Comparativo de Tempos 
#### 20 imagens

| Ambiente | Tempo Total | Observações |
|-----------|--------------|-------------|
| **MATLAB** | 35,17 min | Implementação de referência |
| **Python (com Numba)** | 5,8 min | Redução drástica de tempo com mesmo resultado |
| **Python (sem Numba)** | ≈ 5 horas (6 imagens) | Execução inviável para datasets maiores |

#### 114 imagens

| Ambiente | Tempo Total | Observações |
|-----------|--------------|-------------|
| **MATLAB** | 3.13h  | Implementação de referência |
| **Python (com Numba)** | 30,68 min | Redução drástica de tempo com resultado muito próximo |


> **Resumo:** As otimizações com Numba tornaram a implementação Python não apenas equivalente numericamente ao MATLAB, mas também **significativamente mais rápida**, viabilizando análises em larga escala.

---

## Requisitos
```bash
pip install pandas numpy numba matplotlib
```