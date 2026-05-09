import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

CSV_PATH = r"Resultados_RPperc_continuous\resultados_testes.csv"
OUT_DIR = "RPperc_continuous"

actual_dir = script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(actual_dir, "boxsplot", OUT_DIR )
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(CSV_PATH)

df = df.dropna(subset=['cenario'])
ordem_cenarios = df.groupby('cenario')['acc'].mean().sort_values(ascending=False).index

# =========================================================
# GRÁFICO 1: ACURÁCIA
# =========================================================
plt.figure(figsize=(12, 8)) 
sns.boxplot(data=df, y='cenario', x='acc', order=ordem_cenarios, palette='viridis')
sns.stripplot(data=df, y='cenario', x='acc', order=ordem_cenarios, color='black', alpha=0.5, size=5)

plt.title('Distribuição de Acurácia por Cenário de Ensemble', fontsize=16, pad=15)
plt.xlabel('Acurácia', fontsize=14)
plt.xlim((0.4, 1.0))
plt.ylabel('Cenário Avaliado', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

save_path = os.path.join(save_dir, "acuracia.png")
plt.savefig(save_path, dpi=300) 
plt.show()

# =========================================================
# GRÁFICO 2: F1-MACRO (Melhor para dados desbalanceados)
# =========================================================
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, y='cenario', x='f1_macro', order=ordem_cenarios, palette='magma')
sns.stripplot(data=df, y='cenario', x='f1_macro', order=ordem_cenarios, color='black', alpha=0.5, size=5)

plt.title('Distribuição de F1-Macro por Cenário de Ensemble', fontsize=16, pad=15)
plt.xlabel('F1-Macro', fontsize=14)
plt.xlim((0.4, 1.0))
plt.ylabel('Cenário Avaliado', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

save_path = os.path.join(save_dir, "f1_score.png")
plt.savefig(save_path, dpi=300)
plt.show()