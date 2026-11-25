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

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import torch


TAMANHO_ORIGINAL_DESCRITORES = 180
PREENCHIMENTOS = ("media_fora_cruz", "cinza_fixo")


def ocluir_grupo_descritores(
    img_rec_tensor: torch.Tensor,
    inicio: int,
    fim: int,
    tamanho_original: int = TAMANHO_ORIGINAL_DESCRITORES,
    *,
    preenchimento: str = "cinza_fixo",
) -> torch.Tensor:
    """
    Oclui completamente um grupo de descritores no RecPlot.

    As linhas e colunas recebem 0.0 no tensor normalizado para cinza fixo,
    ou a média por canal dos pixels intactos da própria imagem.
    """
    if img_rec_tensor.dim() == 3:
        img_rec_tensor = img_rec_tensor.unsqueeze(0)

    tamanho_rede = img_rec_tensor.shape[-1]

    if img_rec_tensor.shape[-2] != tamanho_rede:
        raise ValueError(
            "O RecPlot precisa ser quadrado para o mapeamento de grupos: "
            f"recebido {tuple(img_rec_tensor.shape[-2:])}."
        )

    p_in = inicio * tamanho_rede // tamanho_original
    p_out = fim * tamanho_rede // tamanho_original
    if preenchimento == "media_fora_cruz":
        intactos = list(range(p_in)) + list(range(p_out, tamanho_rede))
        if not intactos:
            raise ValueError("O preenchimento pela média precisa de pixels fora da cruz.")
        valor = img_rec_tensor[:, :, intactos, :][:, :, :, intactos].mean(
            dim=(-2, -1), keepdim=True,
        )
    elif preenchimento == "cinza_fixo":
        valor = 0.0
    else:
        raise ValueError(f"Preenchimento desconhecido: {preenchimento}")

    img_oclusa = img_rec_tensor.clone()

    # Todas as LINHAS do grupo
    img_oclusa[:, :, p_in:p_out, :] = valor

    # Todas as COLUNAS do grupo
    img_oclusa[:, :, :, p_in:p_out] = valor

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
    import torch

    from xai.recplot_mapping import definir_bandas_funcoes

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
    from xai.seed_aggregation import carrega_branch_todas_seeds

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
        _, img_rec, label = dataset[sample]
        target_class = int(label)

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


def avaliar_importancia_por_grupo_v2(modelo, img_rec_tensor, y_true: int) -> list[dict]:
    """Mede a variação para a classe predita na imagem intacta e para a verdadeira."""
    import torch

    from xai.recplot_mapping import definir_bandas_funcoes

    entrada = img_rec_tensor.unsqueeze(0).to(next(modelo.parameters()).device)
    grupos = definir_bandas_funcoes()
    cruzes = [
        (grupo, preenchimento, ocluir_grupo_descritores(
            entrada, inicio, fim, preenchimento=preenchimento,
        ))
        for preenchimento in PREENCHIMENTOS
        for grupo, (inicio, fim) in grupos.items()
    ]
    with torch.no_grad():
        lote = torch.cat([entrada, *(img for _, _, img in cruzes)])
        probabilidades = modelo(lote).softmax(dim=1).cpu().numpy()
    if not np.isfinite(probabilidades).all():
        raise ValueError("A oclusão em cruz retornou probabilidades não finitas.")

    base = probabilidades[0]
    y_pred = int(base.argmax())
    prob_predita = float(base[y_pred])
    prob_verdadeira = float(base[y_true])
    linhas = []
    for (grupo, preenchimento, _), oclusa in zip(cruzes, probabilidades[1:]):
        oclusa_predita = float(oclusa[y_pred])
        oclusa_verdadeira = float(oclusa[y_true])
        linhas.append({
            "grupo": grupo,
            "preenchimento": preenchimento,
            "y_true": y_true,
            "y_pred": y_pred,
            "correct": y_pred == y_true,
            "prob_base_predita": prob_predita,
            "prob_oclusa_predita": oclusa_predita,
            "queda_prob_predita": prob_predita - oclusa_predita,
            "prob_base_verdadeira": prob_verdadeira,
            "prob_oclusa_verdadeira": oclusa_verdadeira,
            "queda_prob_verdadeira": prob_verdadeira - oclusa_verdadeira,
        })
    return linhas


def importancia_por_grupo_dataset_v2(
    nome_dataset: str,
    nome_branch: str,
    dataset,
    num_classes: int,
    seeds,
    models_dir: str = "models",
    device: str = "cuda",
) -> pd.DataFrame:
    """Aplica os dois preenchimentos aos nove grupos de todo o teste."""
    from xai.loader import BRANCH_FILES, carregar_modelo

    backbone, tipo_entrada, arquivo = BRANCH_FILES[nome_branch]
    if tipo_entrada != "recplot":
        raise ValueError(f"A oclusão em cruz requer um modelo RecPlot: {nome_branch}")
    if not len(dataset):
        raise ValueError(f"Conjunto de teste vazio para {nome_dataset}.")

    caminhos = {seed: Path(models_dir) / str(seed) / arquivo for seed in seeds}
    ausentes = [str(path) for path in caminhos.values() if not path.is_file()]
    if ausentes:
        raise FileNotFoundError("Checkpoints RecPlot ausentes:\n" + "\n".join(ausentes))

    amostras = []
    for sample in range(len(dataset)):
        _, img_rec, label = dataset[sample]
        amostras.append((img_rec, int(label)))

    linhas = []
    for seed, caminho in caminhos.items():
        modelo = carregar_modelo(backbone, num_classes, tipo_entrada, caminho)
        if modelo is None:
            raise FileNotFoundError(caminho)
        modelo = modelo.to(device).eval()
        for sample, (img_rec, label) in enumerate(amostras):
            linhas.extend(
                {"dataset": nome_dataset, "branch": nome_branch, "seed": seed, "sample": sample, **linha}
                for linha in avaliar_importancia_por_grupo_v2(modelo, img_rec, label)
            )
        del modelo

    nomes_classes = dict(enumerate(dataset.class_names))
    return pd.DataFrame(linhas).assign(
        nome_classe_verdadeira=lambda df: df["y_true"].map(nomes_classes),
        nome_classe_predita=lambda df: df["y_pred"].map(nomes_classes),
    )


def resumir_importancia_v2(detalhes: pd.DataFrame, estratos=()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula estatísticas por semente e agrega as sementes com pesos iguais."""
    dados = pd.concat([
        detalhes.assign(
            alvo=alvo,
            classe_alvo=detalhes[classe],
            nome_classe_alvo=detalhes[f"nome_classe_{alvo}"],
            queda_prob=detalhes[f"queda_prob_{alvo}"],
        )
        for alvo, classe in (("predita", "y_pred"), ("verdadeira", "y_true"))
    ], ignore_index=True)
    dados = dados.assign(
        queda=dados["queda_prob"].gt(0),
        aumento=dados["queda_prob"].lt(0),
        sem_variacao=dados["queda_prob"].eq(0),
    )
    chaves = ["dataset", "branch", "preenchimento", "alvo", "grupo", *estratos]
    por_seed = dados.groupby([*chaves, "seed"], as_index=False).agg(
        media=("queda_prob", "mean"),
        n_observacoes=("sample", "size"),
        n_quedas=("queda", "sum"),
        n_aumentos=("aumento", "sum"),
        n_sem_variacao=("sem_variacao", "sum"),
        proporcao_quedas=("queda", "mean"),
        proporcao_aumentos=("aumento", "mean"),
        proporcao_sem_variacao=("sem_variacao", "mean"),
    )
    resumo = por_seed.groupby(chaves, as_index=False).agg(
        media=("media", "mean"),
        std=("media", "std"),
        n_seeds=("seed", "nunique"),
        n_observacoes=("n_observacoes", "sum"),
        n_quedas=("n_quedas", "sum"),
        n_aumentos=("n_aumentos", "sum"),
        n_sem_variacao=("n_sem_variacao", "sum"),
        proporcao_quedas=("proporcao_quedas", "mean"),
        proporcao_aumentos=("proporcao_aumentos", "mean"),
        proporcao_sem_variacao=("proporcao_sem_variacao", "mean"),
    )
    n_imagens = dados.groupby(chaves, as_index=False).agg(n_imagens=("sample", "nunique"))
    return por_seed, resumo.merge(n_imagens, on=chaves, validate="one_to_one")


def comparar_preenchimentos(resumo: pd.DataFrame) -> pd.DataFrame:
    """Compara a queda média e o ranking dos mesmos grupos nos dois preenchimentos."""
    chaves = ["dataset", "branch", "alvo"]
    tabela = resumo.pivot(index=[*chaves, "grupo"], columns="preenchimento", values="media")
    ranks = tabela.groupby(level=chaves).rank(ascending=False, method="min")
    tabela = tabela.add_prefix("queda_").join(ranks.add_prefix("rank_"))
    tabela["diferenca_media"] = tabela["queda_media_fora_cruz"] - tabela["queda_cinza_fixo"]
    tabela["diferenca_rank"] = (
        tabela["rank_media_fora_cruz"] - tabela["rank_cinza_fixo"]
    ).abs()
    return tabela.reset_index()


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
    import matplotlib.pyplot as plt

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
