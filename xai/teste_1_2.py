import torch
from model.dataset import EnsembleTestDataset
from model.utils import transform_originais, transform_recplot
from xai.loader import carregar_todos_os_branches, obter_camada_alvo
from xai.attributions import gerar_atribuicao, desnormalizar

# 1. Configurações básicas
seed = 7
num_classes = 2 # Exemplo Displasia
models_dir = r"resultados_displasia\models"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Carrega todos os modelos da seed[cite: 1]
branches = carregar_todos_os_branches(seed=seed, num_classes=num_classes, models_dir=models_dir, device=device)

# 3. Carrega o dataset de testes usando a estrutura original[cite: 1]
dataset = EnsembleTestDataset(
    'C:\\Users\\IFTM-ITB\\Desktop\\EnsembleFractal\\datasets\\daniel_tentando\\novo\\RPnew\\teste',
    ['healthy', 'severe'], 
    transform_original=transform_originais, 
    transform_recplot=transform_recplot
)

# Pega uma amostra de teste (Original + RecPlot)
img_original, img_rec, label = dataset[0]
# print(sample)
# img_orig = sample["img_orig"].unsqueeze(0).to(device) # (1, 3, 224, 224)
# img_rec  = sample["img_rec"].unsqueeze(0).to(device)  # (1, 3, 224, 224)
# label = sample["label"]

# 4. Testa a geração de atribuição para o branch RecPlot + MobileNet
modelo_mobilenet_rec = branches["mobilenet_recplot"]
camada = obter_camada_alvo(modelo_mobilenet_rec)
label_int = label.item() if isinstance(label, torch.Tensor) else label

mapa_gradcam = gerar_atribuicao(
    modelo=modelo_mobilenet_rec,
    input_tensor=img_rec,
    target_class=label_int,
    camada_alvo=camada,
    metodo="gradcam"
)

# 5. Obtém a imagem desnormalizada para plotagem[cite: 1]
img_rec_np = desnormalizar(img_rec, tipo="rec")

print(f"Sucesso! Grad-CAM gerado com formato: {mapa_gradcam.shape}")
print(f"Imagem desnormalizada com formato: {img_rec_np.shape}")