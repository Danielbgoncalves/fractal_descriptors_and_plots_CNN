'''
Arquivo voltadopara a explicação dos outputs das redes treinadas e avaliadas nesse diretório
Há dois contextos em que será usado:
    1. Explicar outputs das entradas originais e F-RecPlot
    2. Explicar outputs das entradas RP + GAF + MTF
Como no caso 2 cada canal é uma representação diferente vamos querer uma explicação que venha por 
canais, enquanto no caso 1 a explicação é numa imagem 2D apenas.
'''

import torch
import numpy as np
from PIL import Image
import matplotlib as plt
import torch.nn.functional as F
from captum.attr import IntegratedGradients, visualization as viz

from model.model import carregar_modelo
from model.utils import transform

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def explain_2D(weigths_path, backbone, num_classes, img_path, target_class, classes):
    
    # Carregar Modelo
    model = carregar_modelo(backbone, num_classes=num_classes, path_weights=weigths_path)
    model.to(DEVICE)
    model.eval()

    # Preparar Imagens
    img_pil = Image.open(img_path).convert('RGB')
    input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad = True

    with torch.no_grad:
        output_bruto = model(input_tensor)
        probabilidades = F.softmax(output_bruto, dim=1)[0]
        confianca, classe_pred = torch.max(probabilidades, dim=0)
        
        confianca_pct = confianca.item() * 100
        classe_pred_idx = classe_pred.item()

        classe_predita = classes[classe_pred_idx]
        classe_real = classes[target_class]
        

    # Calcula atribuições com IG
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(input_tensor, target=target_class,
                                       n_step=50, return_convergence_delta=True)
    

    attr_tensor = attributions.squeeze().cpu().detach()
    attr_np = np.transpose(attr_tensor.numpy(), (1,2,0)) 

    # Exatamente igual a normalização do treino/teste
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])

    # Traz de volta o input da rede
    img_tensor = input_tensor.squeeze().cpu().detach().numpy() # (C, H, W)
    img_tensor = np.transpose(img_tensor, (1, 2, 0)) # Transforma para (H, W, C)

    img_fundo = std * img_tensor + mean
    img_fundo = np.clip(img_fundo, 0, 1)

    # Vizualização
    print(f'o modelo previu a classe {classe_predita} com {confianca_pct:.2f}%de confiança e o correto é {classe_real}')

    titulo_grafico = f"Alvo XAI: {classe_real} | Predição: {classe_pred} ({confianca_pct:.2f}%)"

    fig, axis = viz.visualize_image_attr(
        attr_np,                     
        img_fundo,
        method="blended_heat_map",   
        sign="positive",             
        show_colorbar=True,
        title=titulo_grafico,
        use_pyplot=True
    )


if __name__ == "__main__":

    weigths_path = "models/42/mobilenet_originais.pth"
    backbone = 'mobilenet'
    num_classes = 2
    img_path = "seila/completa/na/hora"
    target_class = 1 #severe

    explain_2D(weigths_path, backbone, num_classes, img_path, target_class)
