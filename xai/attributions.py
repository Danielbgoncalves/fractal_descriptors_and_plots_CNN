import torch
import numpy as np
import torch.nn.functional as F
from captum.attr import IntegratedGradients, LayerGradCam, Occlusion

MEAN_STD= {
    "orig": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # ImageNet
    "rec":  ([0.5, 0.5, 0.5],       [0.5, 0.5, 0.5]),       # RecPlot (0.5/0.5)
}

def desnormalizar(tensor_img: torch.Tensor, tipo: str) -> np.ndarray:
    '''
    Inverte a transformação de model/utils.py e converte o Tensor (C, H, W) em 
    array (H, W, C) em [0, 1]
    '''

    if tipo not in MEAN_STD:
        raise ValueError("O parâmetro 'tipo' de ver 'orig' ou 'rec'.")

    mean, std = MEAN_STD[tipo]

    img = tensor_img.clone().detach().cpu() # carambolas

    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]

    img = torch.clamp(img, 0.0, 1.0)

    return img.permute(1, 2, 0).numpy()

def gerar_atribuicao(
        modelo: torch.nn.Module,
        input_tensor: torch.Tensor,
        target_class: int,
        camada_alvo: torch.nn.Module = None,
        metodo: str="gradcam"
) -> torch.Tensor:
    '''
    Gera o mapa de atribuição (regiões de interesse de cada método)
    '''

    modelo.zero_grad()

    device = next(modelo.parameters()).device
    input_tensor = input_tensor.to(device)

    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.requires_grad_(True)
    spatial_size = input_tensor.shape[2:]

    if metodo == "gradcam":
        if camada_alvo is None: raise ValueError("O método Grad-Cam exige que a camada_alvo seja especificada.")
        grad_cam = LayerGradCam(modelo, camada_alvo)
        attr = grad_cam.attribute(input_tensor, target=target_class)
        attr = F.interpolate(attr, size=spatial_size, mode='bilinear', align_corners=False)

    elif metodo == "ig":
        ig = IntegratedGradients(modelo)
        attr = ig.attribute(input_tensor, target=target_class, n_steps=50)


    elif metodo == "occlusion":
        occ = Occlusion(modelo)
        attr = occ.attribute(
            input_tensor, 
            target=target_class, 
            sliding_window_shapes=(3, 16, 16), 
            strides=(3, 8, 8)
        )
    else:
        raise ValueError(f"Método {metodo} não suportado, escolha entre 'gradcam', 'ig', ou 'occlusion'.")

    return attr




