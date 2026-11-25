"""Comparação espacial das atribuições de MobileNetV2 e EfficientNet-B0."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

BRANCHES_POR_REPRESENTACAO = {
    "orig": ("mobilenet_orig", "effnet_orig"),
    "recplot": ("mobilenet_recplot", "effnet_recplot")
}

AMPLITUDE_MINIMA = 1e-12


def _estado_mapa(mapa: np.ndarray) -> str:
    if not np.isfinite(mapa).all():
        return "nao_finito"
    if np.ptp(mapa) < AMPLITUDE_MINIMA:
        return "plano"
    return "valido"


def _normaliza_mapa(attr: Tensor) -> np.ndarray:
    '''
    (1, C, H, W) para heatmap 2D em [0, 1] (soma absoluta dos canais
    e min-max). Mesma lógica usada em xai/sanity_check.py, para comparar 
    o que seria mostrado visualmente numa figura
    '''

    mapa = attr.squeeze(0).detach().abs().sum(dim=0).cpu().numpy()
    estado = _estado_mapa(mapa)
    if estado == "nao_finito":
        return mapa
    if estado == "plano":
        return np.zeros_like(mapa)

    mini, maxi = mapa.min(), mapa.max()
    return (mapa - mini) / (maxi - mini)

def mapa_normalizado(
    modelo: Module,
    input_tensor: Tensor,
    target_class: int,
    metodo: str="gradcam"
) -> np.ndarray:
    '''
    Gera a normalização min-maxde 0 a 1 do mapa de atribuição de
    um branch para uma amostra
    '''
    from xai.attributions import gerar_atribuicao
    from xai.loader import obter_camada_alvo

    camada = obter_camada_alvo(modelo) if metodo == "gradcam" else None
    attr = gerar_atribuicao(modelo, input_tensor, target_class, camada_alvo=camada, metodo=metodo)
    return _normaliza_mapa(attr)

def _mascara_top_k(mapa: np.ndarray, k: int) -> np.ndarray:
    """Distribui a capacidade restante igualmente entre os pixels empatados."""
    pixels = mapa.ravel()
    positivos = pixels > 0
    if positivos.sum() <= k:
        return positivos.astype(float)

    limiar = np.partition(pixels, -k)[-k]
    acima = pixels > limiar
    empatados = pixels == limiar
    pesos = acima.astype(float)
    pesos[empatados] = (k - acima.sum()) / empatados.sum()
    return pesos


def iou_top_k(
    mapa_a: np.ndarray,
    mapa_b: np.ndarray,
    k_percentil: float=10.0
) -> float:
    """IoU dos maiores valores positivos em dois mapas normalizados.

    A capacidade é ceil(k% do total de pixels). Empates positivos no corte
    recebem pesos iguais até completar essa capacidade. A interseção usa o
    mínimo dos pesos por pixel e a união usa o máximo.
    """
    if mapa_a.shape != mapa_b.shape:
        raise ValueError(f"Mapas com formatos diferentes: {mapa_a.shape} vs {mapa_b.shape}")
    if not 0 < k_percentil <= 100:
        raise ValueError("O percentual do IoU deve estar em (0, 100].")
    if _estado_mapa(mapa_a) != "valido" or _estado_mapa(mapa_b) != "valido":
        return np.nan

    k = int(np.ceil(mapa_a.size * k_percentil / 100))
    mascara_a = _mascara_top_k(mapa_a, k)
    mascara_b = _mascara_top_k(mapa_b, k)
    intersecao = np.minimum(mascara_a, mascara_b).sum()
    uniao = np.maximum(mascara_a, mascara_b).sum()
    return float(intersecao / uniao) if uniao else np.nan

def ssim_entre_mapas(
    mapa_a: np.ndarray,
    mapa_b: np.ndarray,
) -> float:
    '''
    Similaridades estrutural entre dois mapas normalizados,
    mais sensível padrões espaciais (bordas e textura) do que
    o IoU
    '''

    from skimage.metrics import structural_similarity

    return float(structural_similarity(mapa_a, mapa_b, data_range=1.0))

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
    entrada = img_orig if tipo_representacao == "orig" else img_rec
    
    if target_class is None:
        target_class = int(label)

    mapa_mobilenet = mapa_normalizado(branches[nome_mobilenet], entrada, target_class, metodo=metodo)
    mapa_effnet = mapa_normalizado(branches[nome_effnet], entrada, target_class, metodo=metodo)
    estado_mobilenet = _estado_mapa(mapa_mobilenet)
    estado_effnet = _estado_mapa(mapa_effnet)
    valido = estado_mobilenet == estado_effnet == "valido"

    return {
        "sample": sample_idx,
        "tipo_representacao": tipo_representacao,
        "target_class": target_class,
        "iou_top_k": iou_top_k(mapa_mobilenet, mapa_effnet, k_percentual),
        "ssim": ssim_entre_mapas(mapa_mobilenet, mapa_effnet) if valido else np.nan,
        "mobilenet_map_status": estado_mobilenet,
        "effnet_map_status": estado_effnet,
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
    cenario_ensemble: str,
    *,
    exigir_todas_amostras: bool = False,
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

    if exigir_todas_amostras and set(df_comparacao["sample"]) != set(predicoes_ensemble["sample"]):
        raise ValueError(f"Cobertura de predições divergente para seed {seed}, {cenario_ensemble}.")
    return df_comparacao.merge(predicoes_ensemble, on="sample", how="inner", validate="one_to_one")

def resumo_concordancia_vs_ensemble(df_cruzado: pd.DataFrame) -> pd.DataFrame:
    '''
    Média e desvio de IoU e SSIM separados por acerto/erro do ensemble
    '''

    return df_cruzado.groupby("acerto_ensemble")[["iou_top_k", "ssim"]].agg(["mean", "std", "count"])


def resumir_comparacao(df: pd.DataFrame, estratos=()) -> pd.DataFrame:
    """Resume as médias por semente e inclui a cobertura por métrica."""
    chaves = ["dataset", "method", "tipo_representacao", *estratos]
    por_seed = df.groupby(chaves + ["seed"], as_index=False).agg(
        iou=("iou_top_k", "mean"),
        ssim=("ssim", "mean"),
    )
    resumo = por_seed.groupby(chaves, as_index=False).agg(
        n_seeds_total=("seed", "nunique"),
        n_seeds_iou_valid=("iou", "count"),
        n_seeds_ssim_valid=("ssim", "count"),
        iou_mean=("iou", "mean"),
        iou_std=("iou", "std"),
        ssim_mean=("ssim", "mean"),
        ssim_std=("ssim", "std"),
    )
    contagens = df.groupby(chaves, as_index=False).agg(
        n_images=("sample", "nunique"),
        n_pairs_total=("sample", "size"),
        n_iou_valid=("iou_top_k", "count"),
        n_ssim_valid=("ssim", "count"),
    )
    return contagens.merge(resumo, on=chaves, validate="one_to_one")
