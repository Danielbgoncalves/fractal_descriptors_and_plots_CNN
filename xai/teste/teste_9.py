from model.dataset import EnsembleTestDataset
from model.utils import transform_originais, transform_recplot
from xai.comparacao_datasets import (
    importancia_por_banda_dataset, resumo_importancia_por_banda,
    comparar_importancia_entre_datasets, figura_comparacao_bandas,
)

dataset_displasia = EnsembleTestDataset("C:\\Users\\IFTM-ITB\\Desktop\\EnsembleFractal\\datasets\\daniel_tentando\\novo\\RPnew\\teste", 
                                        ['healthy', 'severe'],
                                          transform_original=transform_originais, transform_recplot=transform_recplot)
dataset_pulmao = EnsembleTestDataset('caminho/teste/pulmao', ['classe0', 'classe1', 'classe2'],
                                       transform_original=transform_originais, transform_recplot=transform_recplot)

df_displasia = importancia_por_banda_dataset("mobilenet_recplot", dataset_displasia, indices=range(10),
                                               num_classes=2, models_dir="resultados_displasia/models", device="cuda")
df_pulmao = importancia_por_banda_dataset("mobilenet_recplot", dataset_pulmao, indices=range(10),
                                            num_classes=3, models_dir="resultados_pulmao/models", device="cuda")

resumo_displasia = resumo_importancia_por_banda(df_displasia)
resumo_pulmao = resumo_importancia_por_banda(df_pulmao)

tabela = comparar_importancia_entre_datasets(resumo_displasia, resumo_pulmao, "displasia", "pulmão")
print(tabela)
fig = figura_comparacao_bandas(tabela, "displasia", "pulmão")