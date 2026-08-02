'''
Testar a sanidade por randomização em cascata, no trabalho de 
(Adebayo et al., 2018 — "Sanity Checks for Saliency Maps").
Recomendado pela Cláudia ia
Funciona assim:
    Se o mapa de atribuição continuar parecido mesmo depois de 
    destruir os pesos aprendidos de partes cada vez maiores da 
    rede, o método de atribuição não está refletindo o que o modelo 
    aprendeu de verdade. Nesse caso ficamos sem fundamento argumentativo.
'''

import copy 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim
 
from xai.attributions import gerar_atribuicao
from xai.loader import obter_camada_alvo, BRANCH_FILES

def _reinicializar_modulo(modulo: torch.nn.Module) -> None:
    '''
    reinicializa in-place um submódulo como Conv2d, Linear, batnorm, ...
    '''

    if hasattr(modulo, "reset_parameters"):
        modulo.reset_parameters()

def _blocos_do_backbone(modelo: torch.nn.Module):
    '''
    Lista os blocos do modelo, do amis próximo da saída para os mais próximos da entrada
    '''

    blocos = [("classifier", modelo.classifier)]
    for i in reversed(range(len(modelo.features))):
        blocos.append((f"features[{i}]", modelo.features[i]))

    return blocos

def _para_heatmap_2d(attr: torch.Tensor) -> np.ndarray:
    '''
    Converte (1, C, H, W) para 2D [0, 1]
    Para comparar o 2D não os tensores crus
    ''' 

    mapa = attr.squeeze(0).abs().sum(dim=0).detach().cpu().numpy() # uau!
    min, max = mapa.min(), mapa.max()
    if max - min < 1e-12:
        return np.zeros_like(mapa)

    return (mapa - min) / (max - min)

def teste_rand_em_cascata(
        modelo_treinado: torch.nn.Module,
        input_tensor: torch.Tensor,
        target_class: int,
        metodo: str="gradcam",
        obter_camada=obter_camada_alvo
    ):
    '''
    Randomiza os pesos da saida para a entrada e mede, a cada passo,
    a similaridade entre o heatmap atual com o modelo 100%
    '''

    modelo_treinado.eval()
    device = next(modelo_treinado.parameters()).device
    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    def _atribuicao(modelo: torch.nn.Module) -> np.ndarray:
        camada = obter_camada(modelo) if metodo == "gradcam" else None
        attr = gerar_atribuicao(
            modelo, input_tensor, target_class, camada_alvo=camada, metodo=metodo
        )
        return _para_heatmap_2d(attr)

    mapa_ref = _atribuicao(modelo_treinado)

    modelo_cascada = copy.deepcopy(modelo_treinado)
    modelo_cascada.eval()
    blocos = _blocos_do_backbone(modelo_cascada)

    resultados = [
        {"bloco": "nenhum (referencia)", "spearman": 1.0, "ssim": 1.0}
    ]

    for nome_bloco, modulo in blocos:
        for submodulo in modulo.modules():
            _reinicializar_modulo(submodulo)

        mapa_atual = _atribuicao(modelo_cascada)

        rho, _ = spearmanr(mapa_ref.flatten(), mapa_atual.flatten())
        similaridade_ssim = ssim(mapa_ref, mapa_atual, data_range=1.0)

        resultados.append({
            "bloco": nome_bloco,
            "spearman": float(rho) if not np.isnan(rho) else 0.0,
            "ssim": float(similaridade_ssim)
        })

    return resultados

def rodar_sanity_check_todos_branches(
        branches, 
        img_orig_tensor: torch.Tensor,
        img_rec_tensor: torch.Tensor,
        target_class: int,
        metodo: str="gradcam"
    ) -> dict:
    '''
    Roda o teste em cascada para os 4 branches de uma vez, escolhendo 
    entre img_orig_tensor ou img_rec_tensor automaticamente
    '''

    resultados_por_branch = {}

    for nome_branch, modelo in branches.items():
        if modelo is None:
            print(f"AVISO: pulando {nome_branch} (modelo não carregado)")
            continue

        tipo_dataset = BRANCH_FILES[nome_branch][1] # "origianis" ou "recplot"
        entrada = img_orig_tensor if tipo_dataset == "originais" else img_rec_tensor

        resultados_por_branch[nome_branch] = teste_rand_em_cascata(
            modelo, entrada, target_class=target_class, metodo=metodo
        )

    return resultados_por_branch


def plotar_curva_sanidade(
    resultados_por_branch,
    metrica: str = "spearman",
    titulo: str = "Teste de sanidade — randomização em cascata",
) -> Figure:
    """Gráfico de linha: similaridade com a explicação original x quantos
    blocos já foram randomizados, um traço por branch — pronto para
    virar figura do artigo (seção de sanity checks).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
 
    for nome_branch, resultados in resultados_por_branch.items():
        valores = [r[metrica] for r in resultados]
        ax.plot(range(len(valores)), valores, marker="o", label=nome_branch)
 
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Blocos randomizados (0 = modelo treinado; último = rede totalmente aleatória)")
    ax.set_ylabel(metrica.capitalize())
    ax.set_title(titulo)
    ax.legend()
    fig.tight_layout()

    return fig