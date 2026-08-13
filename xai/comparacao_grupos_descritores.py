"""
Comparação entre datasets por grupos de descritores do RecPlot.

A unidade de análise é um grupo de 20 descritores (Métrica x Função),
por exemplo:
    - Minkowski (P): índices 0..19
    - Minkowski (G): índices 20..39
    - ...
    - Manhattan (H): índices 160..179

IMPORTANTE:
A oclusão de um grupo não remove apenas o bloco diagonal 20x20.
Ela remove TODAS as linhas E TODAS as colunas do grupo no RecPlot.
Assim, por exemplo, Minkowski (P) remove:
    - Minkowski(P) x Minkowski(P)
    - Minkowski(P) x qualquer outro grupo
    - qualquer outro grupo x Minkowski(P)

A medida resultante responde à pergunta:
"Quanto a presença deste grupo de descritores no RecPlot influencia
a probabilidade da classe-alvo?"
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from xai.recplot_mapping import definir_bandas_funcoes
from xai.seed_aggregation import carrega_branch_todas_seeds


TAMANHO_ORIGINAL_DESCRITORES = 180


def ocluir_grupo_descritores(
    img_rec_tensor: torch.Tensor,
    inicio: int,
    fim: int,
    tamanho_original: int = TAMANHO_ORIGINAL_DESCRITORES,
) -> torch.Tensor:
    """
    Oclui completamente um grupo de descritores no RecPlot.

    Para um grupo [inicio, fim), todas as linhas e todas as colunas
    correspondentes aos seus descritores são substituídas por 0.0.

    Como o RecPlot foi normalizado com mean=0.5 e std=0.5, 0.0 no
    tensor normalizado corresponde a intensidade 0.5 na imagem original,
    isto é, cinza médio.
    """
    if img_rec_tensor.dim() == 3:
        img_rec_tensor = img_rec_tensor.unsqueeze(0)

    tamanho_rede = img_rec_tensor.shape[-1]

    if img_rec_tensor.shape[-2] != tamanho_rede:
        raise ValueError(
            "O RecPlot precisa ser quadrado para o mapeamento de grupos: "
            f"recebido {tuple(img_rec_tensor.shape[-2:])}."
        )

    fator = tamanho_rede / tamanho_original

    p_in = int(inicio * fator)
    p_out = int(fim * fator)

    img_oclusa = img_rec_tensor.clone()

    # Todas as LINHAS do grupo
    img_oclusa[:, :, p_in:p_out, :] = 0.0

    # Todas as COLUNAS do grupo
    img_oclusa[:, :, :, p_in:p_out] = 0.0

    return img_oclusa


def avaliar_importancia_por_grupo_descritores(
    modelo: torch.nn.Module,
    img_rec_tensor: torch.Tensor,
    target_class: int,
) -> dict:
    """
    Mede a influência de cada grupo de descritores por oclusão em cruz.

    Retorna, para cada grupo:
        prob_base
        prob_oclusa
        queda_prob
    """
    modelo.eval()

    if img_rec_tensor.dim() == 3:
        img_rec_tensor = img_rec_tensor.unsqueeze(0)

    device = next(modelo.parameters()).device
    img_rec_tensor = img_rec_tensor.to(device)

    with torch.no_grad():
        logits_orig = modelo(img_rec_tensor)
        prob_orig = torch.softmax(logits_orig, dim=1)
        prob_base = prob_orig[0, target_class].item()

    resultados = {}

    for nome_grupo, (inicio, fim) in definir_bandas_funcoes().items():
        img_oclusa = ocluir_grupo_descritores(
            img_rec_tensor,
            inicio,
            fim,
        )

        with torch.no_grad():
            logits_oclusa = modelo(img_oclusa)
            prob_oclusa = torch.softmax(logits_oclusa, dim=1)
            prob_masked = prob_oclusa[0, target_class].item()

        queda_prob = prob_base - prob_masked

        resultados[nome_grupo] = {
            "prob_base": prob_base,
            "prob_oclusa": prob_masked,
            "queda_prob": queda_prob,
        }

    return resultados


def importancia_por_grupo_dataset(
    nome_branch: str,
    dataset,
    indices,
    num_classes: int,
    seeds=None,
    models_dir: str = "models",
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Calcula a importância dos 9 grupos para várias amostras e seeds.

    Cada linha representa:
        seed x amostra x grupo
    """
    if "recplot" not in nome_branch.lower():
        raise ValueError(
            "A análise por grupos de descritores só se aplica a branches "
            f"RecPlot, recebido: {nome_branch}"
        )

    if seeds is None:
        seeds = []

    linhas = []

    modelos_por_seed = carrega_branch_todas_seeds(
        nome_branch,
        num_classes,
        seeds=seeds,
        models_dir=models_dir,
        device=device,
    )

    for sample in indices:
        img_orig, img_rec, label = dataset[sample]

        target_class = (
            int(label.item())
            if isinstance(label, torch.Tensor)
            else int(label)
        )

        for seed, modelo in modelos_por_seed.items():
            resultados = avaliar_importancia_por_grupo_descritores(
                modelo,
                img_rec,
                target_class,
            )

            for grupo, valores in resultados.items():
                linhas.append(
                    {
                        "seed": seed,
                        "sample": sample,
                        "target_class": target_class,
                        "grupo": grupo,
                        "prob_base": valores["prob_base"],
                        "prob_oclusa": valores["prob_oclusa"],
                        "queda_prob": valores["queda_prob"],
                    }
                )

    return pd.DataFrame(linhas)


def resumo_importancia_por_grupo(
    df_importancia: pd.DataFrame,
) -> pd.DataFrame:
    """
    Média, desvio e n da queda de probabilidade por grupo.
    """
    return (
        df_importancia
        .groupby("grupo")["queda_prob"]
        .agg(media="mean", std="std", n="count")
        .sort_values("media", ascending=False)
    )


def comparar_importancia_por_grupo_entre_datasets(
    resumo_a: pd.DataFrame,
    resumo_b: pd.DataFrame,
    nome_a: str = "dataset_a",
    nome_b: str = "dataset_b",
) -> pd.DataFrame:
    """
    Compara importância média e ranking dos 9 grupos entre dois datasets.
    """
    a = resumo_a[["media"]].rename(
        columns={"media": f"media_{nome_a}"}
    )
    b = resumo_b[["media"]].rename(
        columns={"media": f"media_{nome_b}"}
    )

    tabela = a.join(b, how="outer")

    tabela[f"rank_{nome_a}"] = tabela[f"media_{nome_a}"].rank(
        ascending=False,
        method="min",
    )
    tabela[f"rank_{nome_b}"] = tabela[f"media_{nome_b}"].rank(
        ascending=False,
        method="min",
    )

    tabela["diferenca_rank"] = (
        tabela[f"rank_{nome_a}"]
        - tabela[f"rank_{nome_b}"]
    ).abs()

    tabela["diferenca_media"] = (
        tabela[f"media_{nome_a}"]
        - tabela[f"media_{nome_b}"]
    )

    return tabela.sort_values(
        "diferenca_rank",
        ascending=False,
    )


def figura_comparacao_grupos(
    tabela_comparacao: pd.DataFrame,
    nome_a: str,
    nome_b: str,
) -> plt.Figure:
    """
    Gráfico lado a lado da importância média de cada grupo.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(tabela_comparacao))
    largura = 0.35

    ax.bar(
        x - largura / 2,
        tabela_comparacao[f"media_{nome_a}"],
        largura,
        label=nome_a,
    )
    ax.bar(
        x + largura / 2,
        tabela_comparacao[f"media_{nome_b}"],
        largura,
        label=nome_b,
    )

    ax.axhline(0.0, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        tabela_comparacao.index,
        rotation=45,
        ha="right",
    )
    ax.set_ylabel(
        "Queda de probabilidade ao ocluir o grupo "
        "(linhas + colunas)"
    )
    ax.set_title(
        f"Influência dos grupos de descritores: "
        f"{nome_a} vs {nome_b}"
    )
    ax.legend()
    fig.tight_layout()

    return fig