import warnings

warnings.filterwarnings("ignore") # Ignora todos os avisos chatos de depreciação de bibliotecas

from .metrics import mostrar_metricas, metrics_to_csv
from .model import carregar_modelo
from .train import train_seeds

from .utils import *
from .dataset import *
from .train import *

# displasia_dataset_path = 'C:\\Users\\IFTM-ITB\\Desktop\\EnsembleFractal\\datasets\\daniel_tentando\\novo\\RPnew\\treino_e_validacao'
# displasia_classes = ['healthy', 'severe'] # 0=healthy, 1=severe
# displasia_test_dir = 'C:\\Users\\IFTM-ITB\\Desktop\\EnsembleFractal\\datasets\\daniel_tentando\\novo\\RPnew\\teste'

lung_dataset_path = 'C:\\Users\\IFTM-ITB\\Desktop\\EnsembleFractal\\datasets\\daniel_tentando\\dataset_lung\\treino_e_validacao'
lung_classes = ['aca_md', 'nor', 'scc_md'] #0=aca_md, 1=nor, 2=scc_md
lung_test_dir = 'C:\\Users\\IFTM-ITB\\Desktop\\EnsembleFractal\\datasets\\daniel_tentando\\dataset_lung\\teste'

OUTPUT_DIR = "resultados_lung/saida4"  

####### Testando com o dataset pulmonar #######
classes = lung_classes
dataset_path = lung_dataset_path
TEST_DIR = lung_test_dir

test_dataset = EnsembleTestDataset(TEST_DIR, classes, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Total de pares de imagens para teste: {len(test_dataset)}")

results = train_seeds(SEEDS, dataset_path, classes, OUTPUT_DIR)

results = {}
num_classes = len(classes)

for seed in SEEDS:
    results[seed] = {}
    results[seed]['mobnet_orig'] = carregar_modelo('mobilenet', num_classes,"originais", f"models/{seed}/mobilenet_originais.pth")
    results[seed]['mobnet_recplot'] = carregar_modelo('mobilenet', num_classes,"recplot",f"models/{seed}/mobilenet_RP_perc.pth")
    results[seed]['effnet_orig'] = carregar_modelo('efficientnet_b0', num_classes,"originais",f"models/{seed}/efficientnet_b0_originais.pth")
    results[seed]['effnet_recplot'] = carregar_modelo('efficientnet_b0', num_classes,"recplot",f"models/{seed}/efficientnet_b0_RP_perc.pth")

metrics_to_csv(SEEDS, results, test_loader, OUTPUT_DIR)