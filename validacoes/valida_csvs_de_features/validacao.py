import pandas as pd
import numpy as np
from scipy.stats import pearsonr, ttest_rel
import matplotlib.pyplot as plt


# Ler arquivos
## Python
p_healf_path = '../codigo-fractal-python/resultados/result_114imgs_healthy_v.csv'
p_severe_path = '../codigo-fractal-python/resultados/result_114imgs_severe_v.csv'

df_p_healthy = pd.read_csv(p_healf_path, sep=',')
df_p_severe = pd.read_csv(p_severe_path, sep=',')

df_python = pd.concat([df_p_healthy, df_p_severe], ignore_index=True)
df_python[[c for c in df_python.columns if c.endswith('Index')]] += 1 # Pra ficar igual ao do matlab (indexado pelo 1)

## MatLab
ml_healthy_path = '..\codigos-fractal-matlab\csv_result_114imgs_healthy\Onlydata-resultado.csv'
ml_severe_path = '..\codigos-fractal-matlab\csv_result_114imgs_severe\Onlydata-resultado.csv'

df_ml_healthy = pd.read_csv(ml_healthy_path, sep=',', header=None)
df_ml_severe = pd.read_csv(ml_severe_path, sep=',', header=None)

df_matlab = pd.concat([df_ml_healthy, df_ml_severe], ignore_index=True)
df_matlab = df_matlab.iloc[:, :-1] # retira a coluna 'classe'

# Eles *precisam* ter mesmo formanto
assert df_matlab.shape == df_python.shape, "Os CSVs têm tamanhos diferentes!"
print(f"Formato dos dados: {df_matlab.shape}")

M = df_matlab.to_numpy(dtype=np.float64)
P = df_python.to_numpy(dtype=np.float64)
n_rows, n_cols = M.shape

erros_abs = np.mean(np.abs(M - P), axis=0)
# média das magnitudes das duas implementações (evita divisão por zero)
mag_media = np.mean(np.abs(M) + np.abs(P), axis=0) / 2
erros_rel = np.where(mag_media > 1e-8, erros_abs / mag_media, np.nan)
rmse = np.sqrt(np.mean((M - P)**2, axis=0))

corr_global = np.corrcoef(P.ravel(), M.ravel())[0, 1]

p_values = np.array([ttest_rel(M[:, j], P[:, j]).pvalue for j in range(n_cols)])
sem_diferenca = np.sum(p_values > 0.05)

print("\n===== RESULTADOS GLOBAIS =====")
print(f"Erro Absoluto Médio Global: {np.nanmean(erros_abs):.6f}")
print(f"Erro Relativo Médio Global: {np.nanmean(erros_rel)*100:.4f}%")
print(f"RMSE Médio Global: {np.nanmean(rmse):.6f}")
print(f"Correlação Média Global: {corr_global:.6f}")
print(f"Colunas sem diferença estatística (p > 0.05): {sem_diferenca}/{n_cols}")

faixas = {
    "< 0.01%": (erros_rel < 0.0001),
    "0.01% - 1%": ((erros_rel >= 0.0001) & (erros_rel < 0.01)),
    "1% - 5%": ((erros_rel >= 0.01) & (erros_rel < 0.05)),
    "5% - 15%": ((erros_rel >= 0.05) & (erros_rel < 0.15)),
}

total = len(erros_rel)
print("\nFaixa de Erro | Descritores | Percentual")
print("--------------|-------------|------------")

for nome, cond in faixas.items():
    qtd = np.sum(cond)
    perc = qtd / total * 100
    print(f"{nome:17} | {qtd:11} | {perc:9.1f}%")


# --- Visualizações ---
plt.figure(figsize=(10, 4))
plt.plot(np.nan_to_num(erros_rel)*100, label='Erro Relativo (%)')
plt.xlabel('Descritor (coluna)')
plt.ylabel('Erro Relativo (%)')
plt.title('Erro Relativo por Coluna - Visão Geral')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(p_values, label='Diferença Estatística', linewidth=1)
plt.axhline(0.05, color='red', linestyle='--', label='p = 0.05 (limite)')
plt.fill_between(range(len(p_values)), 0, 1, where=p_values>0.05, color='green', alpha=0.2, label='Sem Diferença Estatística')
plt.xlabel('Descritor (coluna)')
plt.ylabel('p')
plt.title('Se p < 0.05 indica diferença estatística')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(np.nan_to_num(erros_rel[:360])*100, label='Erro Relativo')
plt.xlabel('Descritor (coluna)')
plt.ylabel('Erro Relativo (%)')
plt.title('Erro Relativo por Coluna - Primeiros 360 descritores (%)')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(np.nan_to_num(erros_rel[360:])*100, label='Erro Relativo')
plt.xlabel('Descritor (coluna)')
plt.ylabel('Erro Relativo (%)')
plt.title('Erro Relativo por Coluna - Últimos 3 descritores (%)')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(5, 5))
plt.scatter(M.flatten(), P.flatten(), s=10, alpha=0.6)
plt.xlabel('MATLAB')
plt.ylabel('Python')
plt.title('Dispersão global: MATLAB x Python')
plt.plot([M.min(), M.max()], [M.min(), M.max()], 'r--')
plt.grid(True)
plt.show()