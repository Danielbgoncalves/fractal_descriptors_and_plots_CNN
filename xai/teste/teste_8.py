from xai.seed_aggregation import mapa_agregado_para_amostra, figura_mapa_agregado, salvar_mapas_agregados
from xai.attributions import desnormalizar
from model.dataset import EnsembleTestDataset


test_dataset = EnsembleTestDataset(
    "C:\\Users\\IFTM-ITB\\Desktop\\EnsembleFractal\\datasets\\daniel_tentando\\novo\\RPnew\\teste", 
    ['healthy', 'severe'],
    dir_originais="data/displasia/test/originais",
    dir_recplots="data/displasia/test/recplots"
)

img_orig, img_rec, label = test_dataset[30]

media, desvio, seeds_usadas = mapa_agregado_para_amostra(
    "mobilenet_recplot", img_orig, img_rec, target_class=label,
    num_classes=2, models_dir="models", metodo="gradcam",
)

fundo = desnormalizar(img_rec, tipo="rec")
fig = figura_mapa_agregado(media, desvio, img_fundo=fundo, titulo="mobilenet_recplot — amostra 42")
salvar_mapas_agregados(media, desvio, seeds_usadas, "resultados_displasia/xai_figuras", "mobilenet_recplot_amostra42")