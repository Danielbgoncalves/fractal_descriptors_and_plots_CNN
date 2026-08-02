from model.dataset import EnsembleTestDataset
from model.utils import transform_originais, transform_recplot, SEEDS
from xai.xai_metrics import xai_metrics_to_csv, mostrar_metricas_xai

dataset = EnsembleTestDataset(
    "C:\\Users\\IFTM-ITB\\Desktop\\EnsembleFractal\\datasets\\daniel_tentando\\novo\\RPnew\\teste", 
    ['healthy', 'severe'],
    transform_original=transform_originais, 
    transform_recplot=transform_recplot
)

df = xai_metrics_to_csv(
    seeds=SEEDS, num_classes=2, models_dir="resultados_displasia/models",
    dataset=dataset, output_dir="resultados_displasia",
    n_amostras=15,   # comece pequeno!
    device="cuda",
)

mostrar_metricas_xai(df)