'''
As 8 seeds são tratadas como variabilidade

Faz uma teste qualitativo, gera mapa médio de atribuição 
para uma mesma amostra + mapa de desvio-padrão
'''

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from model.utils import SEEDS
from xai.loader import carregar_todos_os_branches, obter_camada_alvo, BRANCH_FILES
from xai.attributions import gerar_atribuicao

def carrega_branch_todas_seeds(
    nome_branch: str,
    num_classes: int,
    seeds=None,
    models_dir: str="models",
    device: str="cuda"
):
    if nome_branch not in BRANCH_FILES:
        raise ValueError(f"nome_branch deve ser um de {list(BRANCH_FILES)}, recebido {nome_branch}")

    seeds = seeds if seeds is not None else SEEDS
    modelos_por_seed = {}

    for seed in seeds:
        branches = carregar_todos_os_branches(seed, num_classes, models_dir=models_dir, device=device)
        modelo = branches.get(nome_branch)
        if modelo is None:
            print(f"AVISO: {nome_branch} não carrgadopara seed {seed}, pulando")
            continue
        modelos_por_seed[seed] = modelo

    return modelos_por_seed

def empilhar_atribuicoes_entre_seeds(
    nome_branch: str,
    input_tensor: torch.Tensor,
    target_class: int,
    num_classes: int,
    seeds=None,
    models_dir: str="models",
    metodo: str="gradcam",
    device: str="cuda"
):
    '''
    gera o mapa de atribuicao da MESMA amostra, para o mesmo branch, nas 8 seeds
    e empilha num array (n_seeds, H, W)
    '''

    modelos_por_seed = carrega_branch_todas_seeds(
        nome_branch, num_classes, seeds=seeds, models_dir=models_dir, device=device
    )

    mapas = []
    seeds_usadas = []

    for seed, modelo in modelos_por_seed.items():
        camada = obter_camada_alvo(modelo) if metodo == "gradcam" else None
        attr = gerar_atribuicao(modelo, input_tensor, target_class, camada_alvo=camada, metodo=metodo)

        mapa = attr.detach().cpu().numpy()[0] # só (C, H, W)
        mapa = mapa.mean(axis=0) if mapa.ndim == 3 else mapa

        mapas.append(mapa)
        seeds_usadas.append(seed)

    if not mapas:
        raise RuntimeError(f"Nenhum modelo do branch {nome_branch!r} pôde ser carregado nas seeds fornecidas. Estranho isso...")

    pilha = np.stack(mapas, axis=0)
    return pilha, seeds_usadas

def agregar_mapa_medio(pilha):
    '''
    Mapa médio + desvio padrão entre as seeds empilhadas
    O desvio-padrão é como um padrão de incerteza explicado, 
    regiões com alto desvio são usados por apenas algumas seeds
    '''

    media = pilha.mean(axis=0)
    desvio = pilha.std(axis=0)
    return media, desvio

def normalizar_mapa(mapa:np.ndarray, eps:float=1e-8):
    '''
    Normaliza min-max [0-1] (eu ja num tinha uma função pra isso ?)
    '''
    mini, maxi = mapa.min(), mapa.max()
    return (mapa-mini) / (maxi - mini + eps)

def mapa_agregado_para_amostra(
    nome_branch: str,
    img_orig_tensor: torch.Tensor,
    img_rec_tensor: torch.Tensor,
    target_class: int,
    num_classes: int,
    seeds=None,
    models_dir: str="models",
    metodo: str="gradcam",
    device: str="cuda"
):
    '''
    Função por conveniência: dado um par (img_orig, img_rec)
    como o devolvido por EnsembleTestDataset escolhe automaticamente 
    o branch e normaliza de acordo
    '''

    tipo_dataset = BRANCH_FILES[nome_branch][1]
    input_tensor = img_orig_tensor if tipo_dataset == "originais" else img_rec_tensor

    pilha, seeds_usadas = empilhar_atribuicoes_entre_seeds(
        nome_branch, input_tensor, target_class, num_classes,
        seeds=seeds, models_dir=models_dir, metodo=metodo, device=device
    )

    media, desvio = agregar_mapa_medio(pilha)
    return media, desvio, seeds_usadas

def seed_mais_representativa(
    resultados_teste_csv,
    cenario: str,
    seed_col: str="seed",
    f1_col: str="f1_macro",
    cenario_col: str="cenario"
):
    '''
    Meio que um fallback: retorna a seed cujo f1 é o mais 
    próximo da média das 8 seeds. Para usar quando o mapa 
    agregado ficar muito poluído
    '''

    df = resultados_teste_csv if isinstance(resultados_teste_csv, pd.DataFrame) else pd.read_csv(resultados_teste_csv)
    df = df[df[cenario_col] == cenario]

    if df.empty:
        raise ValueError(f"Nenhuma linha encontradaa para cenario={cenario!r} em resultados_teste_csv")

    media_f1 = df[f1_col].mean()
    df = df.copy()
    df["dist_para_media"] = (df[f1_col] - media_f1).abs()
    linha = df.sort_values("dist_para_media").iloc[0]

    return int(linha[seed_col])

def figura_mapa_agregado(
    media: float,
    desvio: float,
    img_fundo=None,
    titulo=None
) -> plt.figure:
    '''
    Figura com mapa médio de atribuição e mapa de desvio-padrão laod
    a aldo, entre as 8 seeds, para mesma amostra/branch.
    '''

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    pares = [(media, "Mapa médio de atribuição"), (desvio, "Desvio-padrão entre seeds (incerteza)")]

    for ax, (mapa, nome) in zip(axes, pares):
        if img_fundo is not None:
            ax.imshow(img_fundo)

        ax.imshow(normalizar_mapa(mapa), cmap="jet", alpha=0.5 if img_fundo is not None else 1.0)
        ax.set_title(nome)
        ax.axis("off")

    if titulo: fig.suptitle(titulo)
    plt.tight_layout()

    return fig

def salvar_mapas_agregados(
    media: float, 
    desvio: float,
    seeds_usadas,
    output_dir: str,
    nome_base: str
):
    '''
    Salva os mapas agregados .npy e a lista de seeds usadas .csv na
    pasta de figuras (xai_figuras/)
    '''

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, f"{nome_base}_media.npy"), media)
    np.save(os.path.join(output_dir, f"{nome_base}_desvio.npy"), desvio)

    pd.DataFrame({"seed": seeds_usadas}).to_csv(
        os.path.join(output_dir, f"{nome_base}_seeds_usadas.csv"), index=False
    )

