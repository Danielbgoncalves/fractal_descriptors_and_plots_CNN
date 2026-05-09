import warnings

warnings.filterwarnings("ignore") # Ignora todos os avisos chatos de depreciação de bibliotecas

from .metrics import mostrar_metricas, metrics_to_csv
from .model import carregar_modelo
from .train import train_seeds

from .utils import *
from .dataset import *
from .train import *

displasia_dataset_path = '../../datasets/originais_RPperc_continuous/treino_e_validacao'
displasia_classes = ['healthy', 'severe'] # 0=healthy, 1=severe
displasia_test_dir = '../../datasets/originais_RPperc_continuous/teste'


# lung_dataset_path = '../datasets/dataset_pulmao/treino_e_validacao'
# lung_classes = ['aca_md', 'nor', 'scc_md'] #0=aca_md, 1=nor, 2=scc_md
# lung_test_dir = '../datasets/dataset_pulmao/teste'

OUTPUT_DIR = "Resultados_RPperc_continuous"

####### Testando com o dataset displasia #######
classes = displasia_classes
dataset_path = displasia_dataset_path
TEST_DIR = displasia_test_dir

test_dataset = EnsembleTestDataset(TEST_DIR, classes, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Total de pares de imagens para teste: {len(test_dataset)}")

# results = train_seeds(SEEDS, dataset_path, classes, OUTPUT_DIR)

results = {}
num_classes = len(classes)

for seed in SEEDS:
    results[seed] = {}
    results[seed]['mobnet_orig'] = carregar_modelo('mobilenet', num_classes,f"models/{seed}/mobilenet_originais.pth")
    results[seed]['mobnet_recplot'] = carregar_modelo('mobilenet', num_classes,f"models/{seed}/mobilenet_RP_perc.pth")
    results[seed]['effnet_orig'] = carregar_modelo('efficientnet_b0', num_classes,f"models/{seed}/efficientnet_b0_originais.pth")
    results[seed]['effnet_recplot'] = carregar_modelo('efficientnet_b0', num_classes,f"models/{seed}/efficientnet_b0_RP_perc.pth")

metrics_to_csv(SEEDS, results, test_loader, OUTPUT_DIR)