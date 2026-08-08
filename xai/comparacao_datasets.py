'''
Responsável pelo comportamente do XAI entre dois datasets, as regiões mais
importantes do RecPlot udam entre displasia e pulmão ? Quais descritores pesam mais ?
'''

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
 
from xai.recplot_mapping import avaliar_importancia_por_metrica_e_funcao
from xai.seed_aggregation import carrega_branch_todas_seeds

def importancia_por_banda_amostra_todas_seeds(
    nome_branch: str,
    img_rec_tensor: torch.Tensor,
    target_class: int,
    num_class: int,
    seeds=None,
    models_dir: str="models",
    device: str="cuda"
) -> pd.DataFrame:
    '''
    Roda avaliar_importancia_por_metrica_e_funcao() para
    UMA amostra, nas 8 seeds do memso branch e devolve 
    uma linha por seed x {"seed", "banda", "queda_prob"}
    '''

    if "recplot" not in nome_branch:
        raise ValueError(
            f"análise por bandas só se aplica a branches RecPlot, recebido {nome_branch}"
        )

    modelos_por_seed = carrega_branch_todas_seeds(
        nome_branch, num_class, seeds=seeds, models_dir=models_dir, device=device
    )

    linhas = []
    for seed, modelo in modelos_por_seed.items():
        impacto_bandas = avaliar_importancia_por_metrica_e_funcao(modelo, img_rec_tensor, target_class)
        for nome_banda, queda_prob in impacto_bandas.items():
            linhas.append({"seed": seed, "banda": nome_banda, "queda_prob": queda_prob})

    return pd.DataFrame(linhas)

def importancia_por_banda_dataset(
    nome_branch: str,
    dataset,
    indices, 
    num_classes: int,
    seeds=None,
    models_dir: str="models",
    device: str="cuda",
    target_class_fn=None
) -> pd.DataFrame:
    '''
    Agrega importancia_por_banda_amostra_todas_seeds() porvárias
    amostras de UM dataset, quais descritores importam mais, 
    em média, NESSE dataset ?

    target_class_fn é funçã(label) -> int para escolher a classe explicada, 
    por padrão é y_true
    '''

    if target_class_fn is None:
        target_class_fn = lambda label: int(label.item()) if isinstance(label, torch.Tensor) else int(label)

    partes = []
    for i in indices:
        img_orig, img_rec, label = dataset[i]
        target_class = target_class_fn(label)

        df_amostra = importancia_por_banda_amostra_todas_seeds(
            nome_branch, img_rec, target_class, num_classes, 
            seeds=seeds, models_dir=models_dir, device=device
        )
        df_amostra["sample"] = i
        df_amostra["target_class"] = target_class
        partes.append(df_amostra)

    return pd.concat(partes, ignore_index=True)

def resumo_importancia_por_banda(df_importancia: pd.DataFrame)-> pd.DataFrame:
    '''
    Média / desvio / n de queda_prob por banda, através de amostras e 
    seeds
    '''
    return (
        df_importancia.groupby("banda")["queda_prob"]
        .agg(media="mean", std="std", n="count")
        .sort_values("media", ascending=False)
    )


def comparar_importancia_entre_datasets(
    resumo_a: pd.DataFrame,
    resumo_b: pd.DataFrame,
    nome_a: str="dataset_a",
    nome_b: str="dataset_b"
) -> pd.DataFrame:
    '''
    Junta dois resumos de importancia_por_banda (de datasets diferentes)
    numa tabelalado a aldo com ranking

    Se diferenca_rank for alto a banda tem posições diferentes nos kankings
    de importancia entre os datasets, se baixo elas tem importância parecida
    '''

    a = resumo_a[["media"]].rename(columns={"media": f"media_{nome_a}"})
    b = resumo_b[["media"]].rename(columns={"media": f"media_{nome_b}"})

    tabela =  a.join(b, how="outer")
    tabela[f"rank_{nome_a}"] = tabela[f"media_{nome_a}"].rank(ascending=False)
    tabela[f"rank_{nome_b}"] = tabela[f"media_{nome_b}"].rank(ascending=False)
    tabela["diferenca_rank"] = (tabela[f"rank_{nome_a}"] - tabela[f"rank_{nome_b}"]).abs()

    return tabela.sort_values("diferenca_rank", ascending=False)

def figura_comparacao_bandas(
    tabela_comparacao: pd.DataFrame,
    nome_a: str,
    nome_b: str
) -> plt.Figure:
    '''
    Gráfico de barras lado a lado comparando a importância média de 
    cada banda entre os dois datasets
    '''

    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(tabela_comparacao))
    largura = 0.35

    ax.bar(x - largura / 2, tabela_comparacao[f"media_{nome_a}"], largura, label=nome_a)
    ax.bar(x + largura / 2, tabela_comparacao[f"media_{nome_b}"], largura, label=nome_b)
 
    ax.set_xticks(x)
    ax.set_xticklabels(tabela_comparacao.index, rotation=45, ha="right")
    ax.set_ylabel("Queda de probabilidade ao ocluir a banda (importância)")
    ax.set_title(f"Importância por banda de descritor: {nome_a} vs. {nome_b}")
    ax.legend()
    fig.tight_layout()
    
    return fig

