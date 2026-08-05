from xai.seed_aggregation import mapa_agregado_para_amostra, figura_mapa_agregado, salvar_mapas_agregados
from xai.attributions import desnormalizar
from model.dataset import EnsembleTestDataset
from model.utils import transform_originais, transform_recplot



test_dataset = EnsembleTestDataset(
    "C:\\Users\\IFTM-ITB\\Desktop\\EnsembleFractal\\datasets\\daniel_tentando\\novo\\RPnew\\teste", 
    ['healthy', 'severe'],
    transform_original=transform_originais, 
    transform_recplot=transform_recplot
)

img_orig, img_rec, label = test_dataset[30]

media, desvio, seeds_usadas = mapa_agregado_para_amostra(
    "mobilenet_recplot", img_orig, img_rec, target_class=label,
    num_classes=3, models_dir="resultados_pulmao/models", metodo="gradcam",
)

fundo = desnormalizar(img_rec, tipo="rec")
fig = figura_mapa_agregado(media, desvio, img_fundo=fundo, titulo="mobilenet_recplot — amostra 42")
salvar_mapas_agregados(media, desvio, seeds_usadas, "resultados_pulmao/xai_figuras", "mobilenet_recplot_amostra42")

print("Testes concluídos com sucesso!")