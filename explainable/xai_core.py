'''
Arquivo voltado para a explicação dos outputs das redes treinadas e avaliadas nesse diretório
Há dois contextos em que será usado:
    1. Explicar outputs das entradas originais e F-RecPlot
    2. Explicar outputs das entradas RP + GAF + MTF
Como no caso 2 cada canal é uma representação diferente vamos querer uma explicação que venha por 
canais, enquanto no caso 1 a explicação é numa imagem 2D apenas.
'''

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from captum.attr import IntegratedGradients, visualization as viz

from model.model import carregar_modelo
from model.utils import transform

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extrac_explanation_2D(model, img_path, target_class, classes):
    '''
    Roda a imgem na rede e com o IG
    Retorna dados brutos para visualização ou manipulação
    '''

    # Preparar Imagens
    img_pil = Image.open(img_path).convert('RGB')
    input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad = True

    with torch.no_grad():
        output_bruto = model(input_tensor)
        probabilidades = F.softmax(output_bruto, dim=1)[0]
        confianca, classe_pred = torch.max(probabilidades, dim=0)
        
        info_pred = {
            "confianca_pct": confianca.item() * 100,
            "classe_pred": classes[classe_pred.item()],
            "classe_real": classes[target_class],
            "target_class_idx": target_class
        }
        

    # Calcula atribuições com IG
    ig = IntegratedGradients(model)
    attributions, _ = ig.attribute(input_tensor, target=target_class,
                                       n_steps=50, return_convergence_delta=True)
    

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

    return input_tensor, attr_np, img_fundo, info_pred

def calcula_fidelidade_delecao(model, input_tensor, attr_np, target_class, steps=10):
    '''
    Remove prpgrassivamente os pixels os pixels mais importantes para a predição
    segundo o IG e mede a queda de confiança da rede
    '''

    importancia_2d = np.sum(np.abs(attr_np), axis=2)
    importancia_flat = importancia_2d.flatten()

    porcentagens = np.linspace(0, 100, steps+1)
    confianca_registradas = []

    for p in porcentagens:
        if p == 0:
            threshold = np.inf
        else:
            threshold = np.percentile(importancia_2d, 100-p)

        mask_2d = (importancia_2d < threshold).astype(np.float32)
        mask_tensor = torch.tensor(mask_2d).unsqueeze(0).unsqueeze(0).to(DEVICE)

        tensor_perturbado = input_tensor * mask_tensor

        with torch.no_grad():
            output = model(tensor_perturbado)
            probs = F.softmax(output, dim=1)[0]

            confianca_alvo = probs[target_class].item() * 100
            confianca_registradas.append(confianca_alvo)

    return porcentagens, confianca_registradas
    
def visualize_results(attr_np, img_fundo, info_pred, porcentagens, confiancas):
    '''
    Gera figura com 2 gráficos: O mapa de calor e as curvas de deleção
    '''

    fig = plt.figure(figsize=(12,5))

    ax1 = fig.add_subplot(1,2,1)
    titulo_heat = f"XAI: {info_pred['classe_real']} | Pred: {info_pred['classe_pred']} ({info_pred['confianca_pct']}%)"

    viz.visualize_image_attr(
        attr_np,                     
        img_fundo,
        method="blended_heat_map",   
        sign="positive",             
        show_colorbar=True,
        title=titulo_heat,
        plt_fig_axis=(fig, ax1),
        use_pyplot=False
    )

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(porcentagens, confiancas, marker='o', color='red', linestyle='-')
    ax2.set_title('Métricas de Delação (Fidelidade)', fontsize=12)
    ax2.set_xlabel(r'% de pixels mais importantes removidos')
    ax2.set_ylabel(f"Confiança na Classe '{info_pred['classe_real']}' (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    weigths_path = r"C:\Users\IFTM-ITB\Desktop\EnsembleFractal\models\42\mobilenet_originais.pth"
    img_path = r"C:\Users\IFTM-ITB\Desktop\EnsembleFractal\datasets\dataset_displasia\teste\severe\originais\12.tif"
    backbone = 'mobilenet'
    num_classes = 2
    target_class = 1 #severe
    classes = ["healthy", "severe"]

    model = carregar_modelo(backbone, num_classes=2, path_weights=weigths_path)
    model.to(DEVICE)
    model.eval()

    input_tensor, attr_np, img_fundo, info = extrac_explanation_2D(model, img_path, target_class, classes)

    porcentagens, confiancas = calcula_fidelidade_delecao(model, input_tensor, attr_np, target_class, steps=10)

    visualize_results(attr_np, img_fundo, info, porcentagens, confiancas)
