"""Cálculo e agregação da AUC das curvas de similaridade do sanity check."""

import numpy as np
import pandas as pd


def calcular_auc_por_amostra(
    df: pd.DataFrame,
    blocos_por_branch_seed: dict[tuple[str, int], tuple[str, ...]],
) -> pd.DataFrame:
    """Integra somente curvas que cobrem toda a cascata e têm valores finitos."""
    chaves = ["dataset", "method", "branch", "seed", "sample"]
    grades = {
        chave: np.arange(len(blocos)) / (len(blocos) - 1)
        for chave, blocos in blocos_por_branch_seed.items()
    }

    linhas = []
    for valores, grupo in df.groupby(chaves, dropna=False):
        grupo = grupo.sort_values("step")
        linha = dict(zip(chaves, valores))
        chave = (linha["branch"], linha["seed"])
        blocos = blocos_por_branch_seed[chave]
        progressos_esperados = grades[chave]
        linha["n_points_expected"] = len(blocos)
        linha["n_points_observed"] = len(grupo)
        linha["progress_complete"] = bool(
            len(grupo) == len(blocos)
            and np.array_equal(grupo["step"], np.arange(len(blocos)))
            and np.array_equal(grupo["block"], blocos)
            and np.allclose(
                grupo["progress"], progressos_esperados, rtol=0.0, atol=1e-12
            )
        )

        for metrica in ("spearman", "ssim"):
            similaridades = grupo[metrica].to_numpy(dtype=float)
            finitas = np.isfinite(similaridades)
            valida = linha["progress_complete"] and bool(finitas.all())
            auc = np.nan
            if valida:
                # Regra dos trapézios sobre os intervalos da cascata.
                alturas = (similaridades[:-1] + similaridades[1:]) / 2.0
                auc = float(np.sum(alturas * np.diff(progressos_esperados)))
            linha[f"auc_{metrica}"] = auc
            linha[f"auc_{metrica}_valid"] = valida
            linha[f"n_{metrica}_valid"] = int(finitas.sum())
        linhas.append(linha)

    return pd.DataFrame(linhas)


def resumir_auc(auc_amostra: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrega imagens por semente e contabiliza a cobertura de cada métrica."""
    chaves = ["dataset", "method", "branch"]
    auc_amostra = auc_amostra.assign(
        auc_spearman=auc_amostra["auc_spearman"].where(
            np.isfinite(auc_amostra["auc_spearman"])
        ),
        auc_ssim=auc_amostra["auc_ssim"].where(np.isfinite(auc_amostra["auc_ssim"])),
    )
    por_seed = (
        auc_amostra.groupby(chaves + ["seed"], dropna=False)
        .agg(
            auc_spearman=("auc_spearman", "mean"),
            auc_ssim=("auc_ssim", "mean"),
            n_samples_total=("sample", "nunique"),
            n_samples_spearman_auc_valid=("auc_spearman", "count"),
            n_samples_ssim_auc_valid=("auc_ssim", "count"),
        )
        .reset_index()
    )
    resumo = (
        por_seed.groupby(chaves, dropna=False)
        .agg(
            auc_spearman_mean=("auc_spearman", "mean"),
            auc_spearman_std=("auc_spearman", "std"),
            auc_ssim_mean=("auc_ssim", "mean"),
            auc_ssim_std=("auc_ssim", "std"),
            n_seeds_total=("seed", "nunique"),
            n_seeds_spearman_auc_valid=("auc_spearman", "count"),
            n_seeds_ssim_auc_valid=("auc_ssim", "count"),
            n_samples_total=("n_samples_total", "sum"),
            n_samples_spearman_auc_valid=("n_samples_spearman_auc_valid", "sum"),
            n_samples_ssim_auc_valid=("n_samples_ssim_auc_valid", "sum"),
        )
        .reset_index()
    )
    return por_seed, resumo
