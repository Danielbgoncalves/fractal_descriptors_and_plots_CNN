import torch.nn as nn
from .utils import *

#saida2
# def criar_modelo(backbone: str, num_classes: int, pretrained='DEFAULT', dropout_rate=0.2):
#     backbone = backbone.lower()

#     if backbone == "mobilenet":
#         model = models.mobilenet_v2(weights=pretrained)
#         for param in model.parameters():
#             param.requires_grad = False
            
#         in_features = model.classifier[1].in_features
#         model.classifier = nn.Sequential(
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(in_features, num_classes)
#         )

#     elif backbone == "efficientnet_b0":
#         model = models.efficientnet_b0(weights=pretrained)
#         for param in model.parameters():
#             param.requires_grad = False
            
#         in_features = model.classifier[1].in_features
#         model.classifier = nn.Sequential(
#             nn.Dropout(p=dropout_rate, inplace=True),
#             nn.Linear(in_features, num_classes)
#         )

#     else:
#         raise ValueError("backbone deve ser 'mobilenet' ou 'efficientnet_b0'")

#     return model

# import torch.nn as nn
# from torchvision import models

# saida3
# Versão com congelamento mais agressiso, funcionava melhor 
# def criar_modelo(backbone: str, num_classes: int, pretrained='DEFAULT', dropout_rate=0.2):
#     backbone = backbone.lower()

#     if backbone == "mobilenet":
#         model = models.mobilenet_v2(weights=pretrained)
        
#         # 1. Congela TUDO (pesos do ImageNet)
#         for param in model.parameters():
#             param.requires_grad = False
            
#         # 2. Descongela os últimos blocos convolucionais (Fine-tuning)
#         # Bloco 17: Último InvertedResidual block
#         # Bloco 18: Última camada Conv2dNormActivation (expansão para 1280 canais)
#         for param in model.features[17].parameters():
#             param.requires_grad = True
#         for param in model.features[18].parameters():
#             param.requires_grad = True
            
#         # 3. Recria o classificador (já nasce descongelado por padrão)
#         in_features = model.classifier[1].in_features
#         model.classifier = nn.Sequential(
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(in_features, num_classes)
#         )

#     elif backbone == "efficientnet_b0":
#         model = models.efficientnet_b0(weights=pretrained)
        
#         # 1. Congela TUDO (pesos do ImageNet)
#         for param in model.parameters():
#             param.requires_grad = False
            
#         # 2. Descongela os últimos blocos convolucionais (Fine-tuning)
#         # Blocos 7 e 8 contêm as características mais complexas e finais da rede
#         for param in model.features[7].parameters():
#             param.requires_grad = True
#         for param in model.features[8].parameters():
#             param.requires_grad = True
            
#         # 3. Recria o classificador
#         in_features = model.classifier[1].in_features
#         model.classifier = nn.Sequential(
#             nn.Dropout(p=dropout_rate, inplace=True),
#             nn.Linear(in_features, num_classes)
#         )

#     else:
#         raise ValueError("backbone deve ser 'mobilenet' ou 'efficientnet_b0'")

#     return model

#saida4
def criar_modelo(
    backbone: str,
    num_classes: int,
    pretrained='DEFAULT',
    dropout_rate=0.2,
    trainable_blocks=3
):
    backbone = backbone.lower()

    if backbone == "mobilenet":
        model = models.mobilenet_v2(weights=pretrained)

        # Congela tudo
        for param in model.parameters():
            param.requires_grad = False

        # Descongela os últimos blocos
        for param in model.features[-trainable_blocks:].parameters():
            param.requires_grad = True

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

        # Garantir que o classifier seja treinável
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=pretrained)

        # Congela tudo
        for param in model.parameters():
            param.requires_grad = False

        # Descongela os últimos blocos
        for param in model.features[-trainable_blocks:].parameters():
            param.requires_grad = True

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

        # Garantir que o classifier seja treinável
        for param in model.classifier.parameters():
            param.requires_grad = True

    else:
        raise ValueError("backbone deve ser 'mobilenet' ou 'efficientnet_b0'")

    return model

# saida1
# def criar_modelo(backbone: str, num_classes: int, pretrained='DEFAULT'):
#     backbone = backbone.lower()

#     if backbone == "mobilenet":
#         model = models.mobilenet_v2(weights=pretrained)
#         model.classifier[1] = nn.Linear(
#             model.classifier[1].in_features, num_classes
#         )

#     elif backbone == "efficientnet_b0":
#         model = models.efficientnet_b0(weights=pretrained)
#         model.classifier[1] = nn.Linear(
#             model.classifier[1].in_features, num_classes
#         )

#     else:
#         raise ValueError("backbone deve ser 'mobilenet' ou 'efficientnet_b0'")

#     return model

# def criar_modelo(
#     backbone: str,
#     num_classes: int,
#     dataset_type: str, # <--- NOVO PARÂMETRO AQUI
#     pretrained='DEFAULT',
#     dropout_rate=0.2,
#     trainable_blocks=3
# ):
#     backbone = backbone.lower()
    
#     # Verifica se estamos lidando com imagens originais
#     is_originais = (dataset_type == "originais")

#     if backbone == "mobilenet":
#         model = models.mobilenet_v2(weights=pretrained)

#         # Só congela se for o dataset original
#         if is_originais:
#             for param in model.parameters():
#                 param.requires_grad = False
#             # Descongela os últimos blocos
#             for param in model.features[-trainable_blocks:].parameters():
#                 param.requires_grad = True

#         in_features = model.classifier[1].in_features
#         model.classifier = nn.Sequential(
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(in_features, num_classes)
#         )

#         # Garantir que o classifier seja treinável
#         for param in model.classifier.parameters():
#             param.requires_grad = True

#     elif backbone == "efficientnet_b0":
#         model = models.efficientnet_b0(weights=pretrained)

#         # Só congela se for o dataset original
#         if is_originais:
#             for param in model.parameters():
#                 param.requires_grad = False
#             # Descongela os últimos blocos
#             for param in model.features[-trainable_blocks:].parameters():
#                 param.requires_grad = True

#         in_features = model.classifier[1].in_features
#         model.classifier = nn.Sequential(
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(in_features, num_classes)
#         )

#         for param in model.classifier.parameters():
#             param.requires_grad = True

#     else:
#         raise ValueError("backbone deve ser 'mobilenet' ou 'efficientnet_b0'")

#     return model

def carregar_modelo(backbone, num_classes, dataset_type, path_weights):
    #print(f"Carregando {backbone} de {path_weights}...")

    model = criar_modelo(backbone, num_classes, pretrained=None) # Cria o modelo com a arquitetura correta

    # Carrega os pesos treinados
    try:
       model.load_state_dict(torch.load(path_weights, map_location=DEVICE, weights_only=True))
    except FileNotFoundError:
        print(f"ERRO: Arquivo {path_weights} não encontrado! Treine o modelo antes.")
        return None

    model.to(DEVICE)
    model.eval()
    return model