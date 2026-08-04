'''
Comparação espacial das atribuições entre MobileNet e EfficientNet,
para a mesma amostra e representação e cruza com a coluna y_pred das
6 linhas de ensemble real do predicoes.csv

A ideia é responder: Quando duas arquiteturas discordam espacialemente
sobre onde está a evidência (IoU/SSIM baixo) o ensemble tende a acertar ???
'''

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
 
from xai.attributions import gerar_atribuicao
from xai.loader import obter_camada_alvo

BRANCHES_POR_REPRESENTACAO = {
    "orig": ("mobilenet_orig", "effnet_orig"),
    "recplot": ("mobilenet_recplot", "effnet_recplot")
}

def _normaliza_mapa(attr: torch.Tensor) -> np.ndarray:
    '''
    (1, C, H, W) para heatmap 2D em [0, 1] (soma absoluta dos canais
    e min-max). Mesma lógica usada em xai/sanity_check.py, para comparar 
    o que seria mostrado visualmente numa figura
    '''

    mapa = attr.squeeze(0).detach().abs().sum(dim=0).numpy()
    mini, maxi = mapa.min(), mapa.max()
    if maxi - mini < 1e-12:
        return np.zeros_like(mapa)

    return (mapa - mini) / (maxi - mini)

def mapa_normalizado(
    modelo: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    metodo: str="gradcam"
) -> np.ndarray:
    '''
    Gera a normalização min-maxde 0 a 1 do mapa de atribuição de
    um branch para uma amostra
    '''
    camada = obter_camada_alvo(modelo) if metodo == "gradcam" else None
    attr = gerar_atribuicao(modelo, input_tensor, target_class, camada_alvo=camada, metodo=metodo)
    return _normaliza_mapa(attr)

def iou_top_k(
    mapa_a: np.ndarray,
    mapa_b: np.ndarray,
    k_percentil: float=10.0
) -> float:
    '''
    Intersection-over-Union entre os top k% pixels mais importantes
    de camda mapa ja normalizado

    IoU=1: os dois mapas olham exatamente para o mesmo lugar
    IoU=0: os dois mapas possuem sobreposição vazia
    '''

    if mapa_a.shape != mapa_b.shape:
        raise ValueError(f"mapas com chapes diferentes: {mapa_a.shape} vs {mapa_b.shape}")

    limiar_a = np.percentile(mapa_a, 100 - k_percentil)
    limiar_b = np.percentile(mapa_b, 100 - k_percentil)

    mascara_a = mapa_a >= limiar_a
    mascara_b = mapa_b >= limiar_b

    intersecao = np.logical_and(mascara_a, mascara_b).sum()
    uniao = np.logical_or(mascara_a, mascara_b).sum()

    if uniao == 0:
        return 0.0

    return float(intersecao / uniao)

def ssim_entre_mapas(
    mapa_a: np.ndarray,
    mapa_b: np.ndarray,
) -> float:
    '''
    Similaridades estrutural entre dois mapas normalizados,
    mais sensível padrões espaciais (bordas e textura) do que
    o IoU
    '''

    return float(ssim(mapa_a, mapa_b, data_range=1.0))

def comparar_arquiteturas_amostra(
    branches,
    dataset, 
    sample_idx: int,
    tipo_representacao: str,
    target_class=None,
    metodo: str="gradcam",
    k_percentual: float=10.0
):
    '''
    Compara MobileNet vs EfficientNet para UMA amostra, numa única
    representação (original OU recplot, nunca as duas, a ideia é 
    isolar o efeito da ARQUITETURA)

    target_class=None ára usar o y_true da própria amostra (como 
    se respondesse a "o que sustenta a classe correta?). 
    Mas da pra sobrescrever.
    '''

    if tipo_representacao not in BRANCHES_POR_REPRESENTACAO:
        raise ValueError(
            f"tipo_representação deve ser 'orig' ou 'recplot', recebido {tipo_representacao}"
        )

    nome_mobilenet, nome_effnet = BRANCHES_POR_REPRESENTACAO[tipo_representacao]

    img_orig, img_rec, label = dataset[sample_idx]
    entrada = img_orig if tipo_representacao == "orig" else img_orig

    if target_class is None:
        target_class = int(label.item()) if isinstance(label, torch.Tensor) else int(label)

    mapa_mobilenet = mapa_normalizado(branches[nome_mobilenet], entrada, target_class, metodo=metodo)
    mapa_effnet =mapa_normalizado(branches[nome_effnet], entrada, target_class, metodo=metodo)

    return {
        "sample": sample_idx,
        "tipo_representacao": tipo_representacao,
        "target_class": target_class,
        "iou_top_k": iou_top_k(mapa_mobilenet, mapa_effnet, k_percentual),
        "ssim": ssim_entre_mapas(mapa_mobilenet, mapa_effnet)
    }

def comparar_arquiteturas_em_lotes(
    branches,
    dataset,
    indices,
    tipo_representacao: str,
    metodo: str="gradcam",
    k_percentil: float=10.0
) -> pd.DataFrame:
    '''
    Roda comparar_arquiteturas_amostra() para cada lista de 
    amostras e devolve um DataFrame, uma linha por amostra
    '''

    linhas = [
        comparar_arquiteturas_amostra(
            branches, dataset, i, tipo_representacao, metodo=metodo, k_percentual=k_percentil
        )
        for i in indices
    ]

    return pd.DataFrame(linhas)

def cruzar_com_predicoes(
    df_comparacao: pd.DataFrame,
    df_predicoes: pd.DataFrame,
    seed: int,
    cenario_ensemble: str
) -> pd.DataFrame:
    '''
    Junta o resultado de comparar_arquiteturas_em_lotes() com
    a linha do ENSEMBLE (uma das 6 combinações - nunca "auto)
    para cada amostra
    '''

    predicoes_ensemble = df_predicoes[
        (df_predicoes["seed"]==seed) & (df_predicoes["cenario"]==cenario_ensemble)
    ][["sample", "y_true", "y_pred"]].copy()
    predicoes_ensemble["acerto_ensemble"] = predicoes_ensemble["y_pred"] == predicoes_ensemble["y_true"]

    return df_comparacao.merge(predicoes_ensemble, on="sample", how="inner")

def resumo_concordancia_vs_ensemble(df_cruzado: pd.DataFrame) -> pd.DataFrame:
    '''
    Média e desvio de IoU e SSIM separados por acerto/erro do ensemble
    '''

    return df_cruzado.groupby("acerto_ensemble")[["iou_top_k", "ssim"]].agg(["mean", "std", "count"])