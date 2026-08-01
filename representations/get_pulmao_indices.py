import pandas as pd
import numpy as np

SEED = 7 #42
TEST_SIZE = 0.30 #20

df = pd.read_csv("representations/pulmao.csv", sep=";")

# -------------------------
# Extrai informações
# -------------------------

# classe
df["classe"] = df["subfolder"]

# magnificação
df["mag"] = df["image_name"].str.extract(r'_(20x|40x)_')

# id do campo
df["grupo"] = (
    df["image_name"]
      .str.extract(r'_(?:20x|40x)_(\d+)\.jpg')[0]
      .astype(int)
)

# -------------------------
# Split por grupo
# -------------------------

rng = np.random.default_rng(SEED)

teste_idx = []

for classe, dados in df.groupby("classe"):

    grupos = dados["grupo"].unique()

    rng.shuffle(grupos)

    n_test = max(1, round(TEST_SIZE * len(grupos)))

    grupos_teste = grupos[:n_test]

    linhas = dados[dados["grupo"].isin(grupos_teste)].index

    teste_idx.extend(linhas.tolist())

# Ordena
teste_idx = sorted(teste_idx)

# índices amigáveis (1-based)
teste_indices = [i + 1 for i in teste_idx]

print(f"Número de imagens de teste: {len(teste_indices)}")

# -------------------------
# Intervalos para a função
# -------------------------

def compactar(indices):
    if len(indices) == 0:
        return ""

    intervalos = []

    ini = fim = indices[0]

    for x in indices[1:]:
        if x == fim + 1:
            fim = x
        else:
            if ini == fim:
                intervalos.append(str(ini))
            else:
                intervalos.append(f"{ini}-{fim}")
            ini = fim = x

    if ini == fim:
        intervalos.append(str(ini))
    else:
        intervalos.append(f"{ini}-{fim}")

    return ",".join(intervalos)


print("\nString para usar:")

print(compactar(teste_indices))