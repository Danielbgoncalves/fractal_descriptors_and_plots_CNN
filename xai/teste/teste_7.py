import pandas as pd
from model.dataset import EnsembleTestDataset
from model.utils import transform_originais, transform_recplot, SEEDS
from xai.loader import carregar_todos_os_branches
from xai.case_selection import casos_discordancia_entre_branches
from xai.arquiteture_comp import comparar_arquiteturas_em_lote, cruzar_com_predicoes, resumo_concordancia_vs_ensemble

df_predicoes = pd.read_csv("resultados_displasia/predicoes.csv")
dataset = EnsembleTestDataset('caminho/teste', ['healthy', 'severe'],
                               transform_original=transform_originais, transform_recplot=transform_recplot)
branches = carregar_todos_os_branches(seed=7, num_classes=2, models_dir="resultados_displasia/models", device="cuda")

# usa a Etapa 6 pra focar nos casos onde já se sabe que há discordância
indices = casos_discordancia_entre_branches(df_predicoes, seed=7, branch_a="mobilenet_orig", branch_b="effnet_orig")

df_comp = comparar_arquiteturas_em_lote(branches, dataset, indices, tipo_representacao="orig")
df_cruzado = cruzar_com_predicoes(df_comp, df_predicoes, seed=7, cenario_ensemble="MobileNet_Original + EffNet_Original")
print(resumo_concordancia_vs_ensemble(df_cruzado))