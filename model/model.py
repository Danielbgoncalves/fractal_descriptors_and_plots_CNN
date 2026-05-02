import torch.nn as nn

from utils import *

def criar_modelo(backbone: str, num_classes: int, pretrained='DEFAULT'):
    backbone = backbone.lower()

    if backbone == "mobilenet":
        model = models.mobilenet_v2(weights=pretrained)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=pretrained)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    else:
        raise ValueError("backbone deve ser 'mobilenet' ou 'efficientnet_b0'")

    return model


def carregar_modelo(backbone, num_classes, path_weights):
    print(f"Carregando {backbone} de {path_weights}...")
    backbone = backbone.lower()
    if backbone == "mobilenet":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # Carrega os pesos treinados
    try:
        model.load_state_dict(torch.load(path_weights, map_location=DEVICE))
    except FileNotFoundError:
        print(f"ERRO: Arquivo {path_weights} não encontrado! Treine o modelo antes.")
        return None

    model.to(DEVICE)
    model.eval()
    return model