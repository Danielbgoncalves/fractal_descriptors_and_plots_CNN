'''
Casos no limiar de decisão

1. O heatmap desses casos de dúvida é diferente do heatmap dos
    de certeza / acerto confiante? 
2. A explicação fica mais espalhada perto do limiar ? 
3. Um ensemble salvaria a decisão nesses casos ? 

Roda para os dois datasets do mesmo jeito
'''

import os
import numpy as np
import pandas as pd
import quantus
import torch

import matplotlib.pyplot as plt
from xai.attributions import desnormalizar

 
from model.utils import SEEDS
from xai.loader import BRANCH_FILES, carregar_todos_os_branches, obter_camada_alvo
from xai.attributions import gerar_atribuicao
from xai.seed_aggregation import normalizar_mapa
from xai.case_selection import (
    BRANCH_PARA_CENARIO_INDIVIDUAL,
    _adiciona_confianca_e_margem,
    casos_por_confianca,
    casos_por_margem,
    casos_corrigidos_pelo_ensemble,
)
 
try:
    from skimage.metrics import structural_similarity as _ssim
except ImportError:
    _ssim = None

CENARIOS_ENSEMBLE = [
    "MobileNet_Original + MobileNet_RecPlot",
    "MobileNet_Original + EffNet_Original",
    "MobileNet_Original + EffNet_RecPlot",
    "MobileNet_RecPlot + EffNet_Original",
    "MobileNet_RecPlot + EffNet_RecPlot",
    "EffNet_Original + EffNet_RecPlot",
]

_NOME_DISPLAY_BRANCH = {
    nome: cenario_auto.split(" + ")[0]
    for nome, cenario_auto in BRANCH_PARA_CENARIO_INDIVIDUAL.items()
}

def preparar_mapa_ig(attr: torch.Tensor, percentile=90):
    """
    Prepara Integrated Gradients para visualização.

    Mantém apenas os pixels mais importantes segundo
    a magnitude da atribuição.
    """

    mapa = attr.detach().cpu().numpy()[0]

    # [C, H, W] -> importância por pixel
    # magnitude da atribuição, sem cancelar sinais entre canais
    if mapa.ndim == 3:
        mapa = np.abs(mapa).mean(axis=0)
    else:
        mapa = np.abs(mapa)

    # Normaliza
    max_val = mapa.max()
    if max_val > 0:
        mapa = mapa / max_val

    # Mantém somente os pixels acima do percentil
    limiar = np.percentile(mapa, percentile)

    mapa_filtrado = np.where(
        mapa >= limiar,
        mapa,
        0
    )

    return mapa_filtrado

def _heatmap_unico(
    modelo: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    metodo="gradcam"
) -> np.ndarray:
    '''
    Heatmap de uma seed só, sem agregar
    '''

    camada = obter_camada_alvo(modelo) if metodo == "gradcam" else None
    attr = gerar_atribuicao(modelo, input_tensor, target_class, camada_alvo=camada, metodo=metodo)

    if metodo == "ig":
        return preparar_mapa_ig(attr, percentile=90)
    
    mapa = attr.detach().cpu().numpy()[0]
    return mapa.mean(axis=0) if mapa.ndim == 3 else mapa

def similaridade_ssim(
    mapa_a: np.ndarray,
    mapa_b: np.ndarray
) -> float:
    '''
    SSIM entre dois mapas normalizados
    1.0 = Mpaas iguais
    '''

    if _ssim is None:
        raise ImportError("scikit-image não instalado (pip install scikit-image)")

    a = normalizar_mapa(mapa_a)
    b = normalizar_mapa(mapa_b)

    return float(_ssim(a, b, data_range=1.0))

def ensemble_salvaria_amostra(
    df_pred: pd.DataFrame, 
    seed: int, 
    nome_branch: str, 
    sample
) -> bool:
    '''
    True se existir qualquer um dos 6 cenários de ensenble que
    contenha 'nome_branch' e acerte essa amostra, dao que o branch
    sozinho erro.
    '''

    cenario_individual = BRANCH_PARA_CENARIO_INDIVIDUAL[nome_branch]
    display = _NOME_DISPLAY_BRANCH[nome_branch]

    for cenario_ensemble in CENARIOS_ENSEMBLE:
        if display not in cenario_ensemble.split(" + "):
            continue

        corrigidos = casos_corrigidos_pelo_ensemble(df_pred, seed, cenario_individual, cenario_ensemble)
        if sample in corrigidos:
            return True

    return False

def analisar_limiar_de_decisao( # ai q preguiça de por tipos!
    predicoes_csv,
    dataset_obj,
    nome_dataset,
    num_classes,
    output_dir,
    seeds=None,
    branches=None,
    margem_max=0.1,
    n_confiantes=15,
    models_dir="models",
    metodo="gradcam",
    device="cuda",   
):
    '''
    Para cada seed x branch:
    1. seleciona 3 grupos a partir de predicoes.csv:
        - "fronteira": margem <= margem_max, acerte ou erro
        - "acerto_confiante": os n_confiantes casos de maior 
            confiança que acertaram
        - "erro_confiante": os n_confiantes casos de amior 
            confiança que erraram
    2. gera o heatmap de cada amostra selecionada (uma seed por vez) e calcula complexidade
    3. compara cada heatmap da fronteira/errou_confiante, via SSIM, contra 
        heatmap MÉDIO do grupo acerto_confiante do mesmo branch/seed
    4. para as amostras erradas marca se algum ensemble salvaria a predição
    
    PS: dataset_obj: EnsembleTestDataset já instanciado (mesmo de run.py) —
    usado pra reabrir a imagem pelo índice `sample` do predicoes.csv
    '''

    seeds = seeds if seeds is not None else SEEDS
    branches = branches if branches is not None else list(BRANCH_FILES)
 
    df_pred = pd.read_csv(predicoes_csv)
    linhas = []
 
    for seed in seeds:
        print(f"[limiar] {nome_dataset} seed {seed}")
        branches_modelos = carregar_todos_os_branches(seed, num_classes, models_dir=models_dir, device=device)
 
        for nome_branch in branches:
            modelo = branches_modelos.get(nome_branch)
            if modelo is None:
                print(f"    AVISO: {nome_branch} não carregado para seed {seed}, pulando")
                continue
 
            cenario = BRANCH_PARA_CENARIO_INDIVIDUAL[nome_branch]
            tipo_dataset = BRANCH_FILES[nome_branch][1]  # "originais" ou "recplot"
 
            df_fronteira = casos_por_margem(df_pred, seed, cenario, margem_max=margem_max)
            samples_fronteira = df_fronteira["sample"].tolist()
            samples_acerto_conf = casos_por_confianca(df_pred, seed, cenario, tipo="acerto_confiante", n=n_confiantes)
            samples_erro_conf = casos_por_confianca(df_pred, seed, cenario, tipo="erro_confiante", n=n_confiantes)
 
            grupos = {
                "fronteira": samples_fronteira,
                "acerto_confiante": samples_acerto_conf,
                "erro_confiante": samples_erro_conf,
            }
            amostras_unicas = sorted(set(samples_fronteira) | set(samples_acerto_conf) | set(samples_erro_conf))
 
            # heatmap de cada amostra única (evita recalcular a mesma
            # amostra 2x se ela aparecer em mais de um grupo)
            cache = {}
            for sample in amostras_unicas:
                linha = df_pred[(df_pred["seed"] == seed) & (df_pred["cenario"] == cenario) & (df_pred["sample"] == sample)]
                if linha.empty:
                    continue
                y_pred = int(linha.iloc[0]["y_pred"])
                img_orig, img_rec, label = dataset_obj[sample]
                input_tensor = img_orig if tipo_dataset == "originais" else img_rec
                mapa = _heatmap_unico(modelo, input_tensor, y_pred, metodo=metodo)
                cache[sample] = (mapa, input_tensor, y_pred)
 
            # referência: heatmap médio do grupo acerto_confiante
            mapas_ref = [cache[s][0] for s in samples_acerto_conf if s in cache]
            mapa_referencia = np.mean(mapas_ref, axis=0) if mapas_ref else None
 
            for categoria, samples in grupos.items():
                for sample in samples:
                    if sample not in cache:
                        continue
                    mapa, input_tensor, y_pred = cache[sample]
 
                    linha_pred = df_pred[(df_pred["seed"] == seed) & (df_pred["cenario"] == cenario) & (df_pred["sample"] == sample)]
                    linha_pred = _adiciona_confianca_e_margem(linha_pred).iloc[0]
                    errou = bool(linha_pred["y_pred"] != linha_pred["y_true"])
 
                    x_batch = input_tensor.unsqueeze(0).numpy()
                    y_batch = np.array([y_pred])
                    a_batch = mapa[None, None, :, :]
 
                    try:
                        complexidade = float(quantus.Sparseness(
                            return_aggregate=False, display_progressbar=False, disable_warnings=True,
                        )(model=modelo, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch,
                          channel_first=True, device=str(device))[0])
                    except Exception:
                        complexidade = np.nan
 
                    ssim_vs_confiante = np.nan
                    if mapa_referencia is not None and categoria != "acerto_confiante" and _ssim is not None:
                        ssim_vs_confiante = similaridade_ssim(mapa, mapa_referencia)
 
                    ensemble_salva = (
                        ensemble_salvaria_amostra(df_pred, seed, nome_branch, sample)
                        if errou else False
                    )
 
                    linhas.append({
                        "dataset": nome_dataset,
                        "seed": seed,
                        "branch": nome_branch,
                        "sample": int(sample),
                        "categoria": categoria,
                        "confianca": float(linha_pred["confianca"]),
                        "margem": float(linha_pred["margem"]),
                        "acertou": not errou,
                        "complexidade_heatmap": complexidade,
                        "ssim_vs_grupo_confiante": ssim_vs_confiante,
                        "ensemble_salvaria": ensemble_salva,
                    })
 
    df_resultado = pd.DataFrame(linhas)
    os.makedirs(output_dir, exist_ok=True)
    df_resultado.to_csv(os.path.join(output_dir, "limiar_analysis.csv"), index=False)
 
    df_resumo = resumir_limiar(df_resultado)
    df_resumo.to_csv(os.path.join(output_dir, "limiar_analysis_summary.csv"), index=False)
 
    return df_resultado, df_resumo

def resumir_limiar(df: pd.DataFrame):
    '''
    Agregar limiar_analysis.csv por dataset xbranch x categoria:
    media/std de complexidade e de SSIM contra grupo confiante.
    % de acerto e % de casos em que o ensemble salavria a decisao
    '''

    return (
        df.groupby(["dataset", "branch", "categoria"])
        .agg(
            n=("sample", "count"),
            complexidade_media=("complexidade_heatmap", "mean"),
            complexidade_std=("complexidade_heatmap", "std"),
            ssim_medio_vs_confiante=("ssim_vs_grupo_confiante", "mean"),
            ssim_std_vs_confiante=("ssim_vs_grupo_confiante", "std"),
            pct_acerto=("acertou", "mean"),
            pct_ensemble_salvaria=("ensemble_salvaria", "mean"),
        )
        .reset_index()
    )

def gerar_figura_exemplos(
    dataset_obj,
    df_resultado,
    nome_branch,
    seed,
    n_por_categoria=3,
    metodo="gradcam",
    models_dir="models",
    num_classes=2,
    device="cuda", 
) -> plt.Figure:
    '''
    Até n_por_categoria heatmaps lado a aldo de cada uma das 3 
    categorias (fronteira / acerto_confiante / erro_confiante), por
    mesmo branch/seed.
    Responde visualmente: "o heatmap muda perto do limiar?"
    '''

    tipo_dataset = BRANCH_FILES[nome_branch][1]
    tipo_norm = "orig" if tipo_dataset == "originais" else "rec"
 
    branches_modelos = carregar_todos_os_branches(seed, num_classes, models_dir=models_dir, device=device)
    modelo = branches_modelos[nome_branch]
 
    categorias = ["acerto_confiante", "fronteira", "erro_confiante"]
    subset = df_resultado[(df_resultado["branch"] == nome_branch) & (df_resultado["seed"] == seed)]
 
    fig, axes = plt.subplots(len(categorias), n_por_categoria, figsize=(4 * n_por_categoria, 4 * len(categorias)))
 
    for i, categoria in enumerate(categorias):
        amostras = subset[subset["categoria"] == categoria].head(n_por_categoria)
        for j in range(n_por_categoria):
            ax = axes[i, j]
            if j >= len(amostras):
                ax.axis("off")
                continue
 
            linha = amostras.iloc[j]
            sample = int(linha["sample"])
            img_orig, img_rec, label = dataset_obj[sample]
            input_tensor = img_orig if tipo_dataset == "originais" else img_rec
 
            mapa = _heatmap_unico(modelo, input_tensor, int(label), metodo=metodo)
            fundo = desnormalizar(input_tensor, tipo=tipo_norm)
 
            ax.imshow(fundo)
            ax.imshow(normalizar_mapa(mapa), cmap="jet", alpha=0.5)
            ax.set_title(f"{categoria}\nconf={linha['confianca']:.2f} margem={linha['margem']:.2f}", fontsize=9)
            ax.axis("off")
 
    fig.suptitle(f"{nome_branch} — seed {seed} — fronteira vs. certeza")
    plt.tight_layout()
    return fig


    