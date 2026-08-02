from ..model import carregar_modelo

import os
import torch

BRANCH_FILES = {
    "mobilenet_orig":     ("mobilenet",        "originais", "mobilenet_originais.pth"),
    "mobilenet_recplot":  ("mobilenet",        "recplot",   "mobilenet_RP_perc.pth"),
    "effnet_orig":        ("efficientnet_b0",  "originais", "efficientnet_b0_originais.pth"),
    "effnet_recplot":     ("efficientnet_b0",  "recplot",   "efficientnet_b0_RP_perc.pth"),
}

def carregar_todos_os_branches(seed: int, num_classes: int, models_dir: str="models", device: str="cuda"):
    '''
    Carrega os 4 modelos (branches) já treinados de uma seed específica

    Retorna:
        dict: dicionário com os modelos em eval por nome
    '''
    branches = {}
    for nome, (backbone, tipo_dataset, nome_arquivo) in BRANCH_FILES.items():
        caminho = os.path.join(backbone, str(seed), nome_arquivo)

        if not caminho: raise FileNotFoundError(f"Pesos não encontrados em {caminho}")

        modelo = carregar_modelo(backbone, num_classes, tipo_dataset, caminho)
        modelo = modelo.to(device)
        modelo.eval()
        branches[nome] = modelo 

    return branches

def obter_camda_alvo(modelo: torch.nn.Module) -> torch.nn.Module:
    '''
    Retorna a última camada convolucional dos modelos 
    '''

    return modelo.features[-1]
