"""Relação entre margem preditiva e mapas Grad-CAM nos RecPlots de teste."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from xai.case_selection import (
    CENARIO_INDIVIDUAL_PARA_BRANCH,
    _adiciona_confianca_e_margem,
    _colunas_de_probabilidade,
)


BRANCHES = ("mobilenet_recplot", "effnet_recplot")
CORTES_MARGEM = (0.05, 0.10, 0.20)
CORTE_PRINCIPAL = 0.10
AMPLITUDE_MINIMA = 1e-12


def preparar_predicoes(df: pd.DataFrame, nome_dataset: str, seeds=None) -> pd.DataFrame:
    """Seleciona e valida a grade completa de predições individuais RecPlot."""
    previsoes = df.assign(branch=df["cenario"].map(CENARIO_INDIVIDUAL_PARA_BRANCH))
    previsoes = previsoes.loc[previsoes["branch"].isin(BRANCHES)].copy()
    seeds = sorted(df["seed"].unique()) if seeds is None else sorted(set(seeds))
    previsoes = previsoes.loc[previsoes["seed"].isin(seeds)]
    chaves = ["branch", "seed", "sample"]
    amostras = sorted(df.loc[df["seed"].isin(seeds), "sample"].unique())
    grade = set(product(BRANCHES, seeds, amostras))
    observadas = set(previsoes[chaves].itertuples(index=False, name=None))
    if not grade or previsoes.duplicated(chaves).any() or observadas != grade:
        raise ValueError("As predições RecPlot devem cobrir uma vez cada modelo, seed e imagem.")
    if amostras != list(range(len(amostras))):
        raise ValueError("Os índices das imagens de teste devem ser consecutivos a partir de zero.")

    prob = previsoes[_colunas_de_probabilidade(previsoes)].to_numpy(dtype=float)
    if (
        prob.shape[1] < 2
        or not np.isfinite(prob).all()
        or (prob < 0).any()
        or not np.allclose(prob.sum(axis=1), 1)
    ):
        raise ValueError("Probabilidades de classe ausentes ou inválidas em predicoes.csv.")
    rotulos = previsoes[["y_true", "y_pred"]].to_numpy()
    if (
        not np.issubdtype(rotulos.dtype, np.integer)
        or (rotulos < 0).any()
        or (rotulos >= prob.shape[1]).any()
        or not np.allclose(prob[np.arange(len(prob)), rotulos[:, 1]], prob.max(axis=1))
        or previsoes.groupby("sample")["y_true"].nunique().ne(1).any()
    ):
        raise ValueError("Classes verdadeiras ou preditas inconsistentes em predicoes.csv.")

    previsoes = _adiciona_confianca_e_margem(previsoes)
    previsoes = previsoes.assign(
        dataset=nome_dataset,
        correct=previsoes["y_true"].eq(previsoes["y_pred"]),
        p_top1=previsoes["confianca"],
        p_top2=previsoes["confianca"] - previsoes["margem"],
    ).rename(columns={"margem": "margin"})
    colunas = ["dataset", *chaves, "y_true", "y_pred", "correct", "p_top1", "p_top2", "margin"]
    return previsoes[colunas].sort_values(chaves).reset_index(drop=True)


def _pacientes_por_amostra(caminho, amostras):
    if caminho is None:
        return {}

    tabela = pd.read_csv(caminho, dtype={"patient_id": "string"})
    ids = tabela["patient_id"].str.strip()
    if tabela["sample"].duplicated().any() or set(tabela["sample"]) != set(amostras):
        raise ValueError("O CSV de pacientes deve associar exatamente um ID a cada sample.")
    if ids.isna().any() or ids.eq("").any():
        raise ValueError("O CSV de pacientes contém IDs ausentes.")
    return dict(zip(tabela["sample"], ids))


def _normalizar_gradcam(mapa):
    mapa = np.asarray(mapa, dtype=np.float32)
    if not np.isfinite(mapa).all():
        return None, "mapa_nao_finito"
    minimo, maximo = mapa.min(), mapa.max()
    if maximo - minimo < AMPLITUDE_MINIMA:
        return None, "mapa_plano"
    return (mapa - minimo) / (maximo - minimo), "valido"


def _prototipo(amostra, classe, grupo, mapas, pacientes):
    referencias = grupo.loc[grupo["correct"] & grupo["y_pred"].eq(classe), "sample"]
    elegiveis = [
        indice for indice in referencias
        if indice in mapas and indice != amostra
        and (not pacientes or pacientes[indice] != pacientes[amostra])
    ]
    if not elegiveis:
        return None, 0
    return np.mean([mapas[indice] for indice in elegiveis], axis=0), len(elegiveis)


def comparar_com_prototipos(grupo, mapas_brutos, pacientes, calcular_ssim):
    """Calcula SSIM fora da referência para um modelo e uma seed."""
    mapas = {}
    estados = {}
    for sample, atribuicao in mapas_brutos.items():
        mapa, estados[sample] = _normalizar_gradcam(atribuicao)
        if mapa is not None:
            mapas[sample] = mapa

    resultados = []
    for row in grupo.itertuples(index=False):
        referencia, n_ref = _prototipo(row.sample, row.y_pred, grupo, mapas, pacientes)
        estado = estados[row.sample]
        valor = np.nan
        if estado == "valido":
            if referencia is None:
                estado = "sem_referencia"
            elif np.ptp(referencia) < AMPLITUDE_MINIMA:
                estado = "referencia_plana"
            else:
                valor = float(calcular_ssim(mapas[row.sample], referencia))
                if not np.isfinite(valor):
                    valor, estado = np.nan, "ssim_nao_finito"
        resultados.append((n_ref, valor, estado))
    metricas = pd.DataFrame(
        resultados, columns=["n_reference_maps", "ssim", "ssim_status"], index=grupo.index
    )
    metricas["reference_excludes_patient"] = bool(pacientes)
    return pd.concat([grupo, metricas], axis=1), mapas


def resumir_grupos(df: pd.DataFrame) -> pd.DataFrame:
    """Conta os quatro grupos para cada corte, incluindo células vazias."""
    chaves = ["dataset", "branch", "correct", "near_boundary"]
    grade = pd.MultiIndex.from_tuples(
        [(dataset, branch, acerto, margem)
         for dataset, branch in df[["dataset", "branch"]].drop_duplicates().itertuples(index=False, name=None)
         for acerto, margem in product((True, False), repeat=2)],
        names=chaves,
    )
    agregacoes = {
        "n_observations": ("sample", "size"),
        "n_images": ("sample", "nunique"),
        "n_seeds": ("seed", "nunique"),
    }
    if "ssim" in df:
        agregacoes.update(
            n_ssim_valid=("ssim", "count"),
            ssim_median=("ssim", "median"),
            ssim_q1=("ssim", lambda values: values.quantile(0.25)),
            ssim_q3=("ssim", lambda values: values.quantile(0.75)),
        )

    tabelas = []
    for corte in CORTES_MARGEM:
        grupos = df.assign(near_boundary=df["margin"].le(corte)).groupby(chaves)
        resumo = grupos.agg(**agregacoes).reindex(grade).reset_index()
        contagens = [col for col in agregacoes if col.startswith("n_")]
        resumo[contagens] = resumo[contagens].fillna(0).astype(int)
        resumo.insert(2, "margin_cutoff", corte)
        tabelas.append(resumo)
    return pd.concat(tabelas, ignore_index=True)


def resumir_correlacoes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula Spearman por seed, no conjunto e dentro de cada classe predita."""
    chaves = ["dataset", "branch", "seed"]
    dados = pd.concat(
        [df.assign(predicted_class="all"), df.assign(predicted_class=df["y_pred"].astype(str))]
    )
    registros = []
    for chave, grupo in dados.groupby(chaves + ["predicted_class"]):
        pares = grupo[["margin", "ssim"]].dropna()
        rho = np.nan
        if len(pares) >= 3 and pares.nunique().gt(1).all():
            ranks = pares.rank()
            rho = ranks["margin"].corr(ranks["ssim"])
        registros.append((*chave, len(grupo), len(pares), rho))

    por_seed = pd.DataFrame(
        registros, columns=[*chaves, "predicted_class", "n_observations", "n_ssim_valid", "spearman"]
    )
    resumo = por_seed.groupby(["dataset", "branch", "predicted_class"], as_index=False).agg(
        n_seeds_total=("seed", "nunique"),
        n_seeds_valid=("spearman", "count"),
        spearman_median=("spearman", "median"),
        spearman_q1=("spearman", lambda values: values.quantile(0.25)),
        spearman_q3=("spearman", lambda values: values.quantile(0.75)),
    )
    return por_seed, resumo


def _predicoes_da_inferencia(grupo, probabilidades):
    """Vincula a análise às predições atuais e preserva a referência histórica."""
    if (
        probabilidades.ndim != 2
        or probabilidades.shape[0] != len(grupo)
        or probabilidades.shape[1] < 2
        or not np.isfinite(probabilidades).all()
    ):
        raise ValueError("A inferência RecPlot retornou probabilidades inválidas.")

    topo = np.sort(probabilidades, axis=1)[:, -2:][:, ::-1]
    classes = probabilidades.argmax(axis=1)
    top2_divergente = ~np.isclose(
        topo, grupo[["p_top1", "p_top2"]].to_numpy(), rtol=1e-3, atol=1e-5
    ).all(axis=1)
    historico = grupo.rename(columns={
        "y_pred": "y_pred_csv",
        "correct": "correct_csv",
        "p_top1": "p_top1_csv",
        "p_top2": "p_top2_csv",
        "margin": "margin_csv",
    })
    return historico.assign(
        y_pred=classes,
        correct=grupo["y_true"].eq(classes),
        p_top1=topo[:, 0],
        p_top2=topo[:, 1],
        margin=topo[:, 0] - topo[:, 1],
        class_changed=grupo["y_pred"].ne(classes),
        top2_outside_tolerance=top2_divergente,
    )


def _inferir_e_gerar_mapas(grupo, modelo, entradas, device):
    import torch

    from xai.attributions import gerar_atribuicao
    from xai.loader import obter_camada_alvo

    entradas_grupo = entradas[grupo["sample"].tolist()]
    with torch.no_grad():
        probabilidades = torch.cat([
            modelo(lote.to(device)).softmax(dim=1).cpu()
            for lote in entradas_grupo.split(16)
        ]).numpy()
    atuais = _predicoes_da_inferencia(grupo, probabilidades)
    n_classes = atuais["class_changed"].sum()
    n_top2 = atuais["top2_outside_tolerance"].sum()
    if n_classes or n_top2:
        print(
            f"[LIMIAR V2][AVISO] {grupo['dataset'].iat[0]} | "
            f"{grupo['branch'].iat[0]} | seed {grupo['seed'].iat[0]} | "
            f"{n_classes}/{len(grupo)} classes e {n_top2}/{len(grupo)} pares de "
            "probabilidades divergem do CSV. A análise usa a inferência atual."
        )

    camada = obter_camada_alvo(modelo)
    mapas = {
        row.sample: gerar_atribuicao(
            modelo, entradas[row.sample], row.y_pred, camada_alvo=camada, metodo="gradcam"
        )[0, 0].detach().cpu().numpy()
        for row in atuais.itertuples(index=False)
    }
    return atuais, mapas


def _selecionar_exemplos(detalhes, mapas_por_grupo, pacientes):
    """Escolhe uma observação próxima do SSIM mediano em cada grupo."""
    validos = detalhes.dropna(subset=["ssim"]).assign(
        near_boundary=lambda df: df["margin"].le(CORTE_PRINCIPAL)
    )
    exemplos = {}
    por_seed = detalhes.groupby(["branch", "seed"])
    for chave, grupo in validos.groupby(["branch", "correct", "near_boundary"]):
        distancia = (grupo["ssim"] - grupo["ssim"].median()).abs()
        row = grupo.loc[distancia.idxmin()]
        chave_seed = row["branch"], row["seed"]
        mapas = mapas_por_grupo[chave_seed]
        referencia, _ = _prototipo(
            row["sample"], row["y_pred"], por_seed.get_group(chave_seed), mapas, pacientes
        )
        exemplos[chave] = row, mapas[row["sample"]], referencia
    return exemplos


def executar_limiar_v2(
    predicoes_csv, dataset, nome_dataset, output_dir, models_dir, *,
    seeds=None, device="cpu", pacientes_csv=None,
):
    """Gera as tabelas e figuras de Grad-CAM para as duas configurações RecPlot."""
    import torch

    from model.model import carregar_modelo
    from xai.comparacao_arquiteturas import ssim_entre_mapas
    from xai.loader import BRANCH_FILES
    from xai.threshold_analysis_v2_plots import salvar_figuras

    predicoes = preparar_predicoes(pd.read_csv(predicoes_csv), nome_dataset, seeds)
    amostras = predicoes.drop_duplicates("sample").sort_values("sample")
    if len(amostras) != len(dataset):
        raise ValueError("O dataset deve conter exatamente as imagens pareadas de predicoes.csv.")
    pacientes = _pacientes_por_amostra(pacientes_csv, amostras["sample"])
    grupos = predicoes.groupby(["branch", "seed"])
    caminhos = {
        (branch, seed): Path(models_dir) / str(seed) / BRANCH_FILES[branch][2]
        for branch, seed in grupos.groups
    }
    ausentes = [str(path) for path in caminhos.values() if not path.is_file()]
    if ausentes:
        raise FileNotFoundError("Checkpoints RecPlot ausentes:\n" + "\n".join(ausentes))

    entradas = []
    for row in amostras.itertuples(index=False):
        _, recplot, label = dataset[row.sample]
        if int(label) != row.y_true:
            raise ValueError(f"Rótulo divergente do CSV para sample={row.sample}.")
        entradas.append(recplot)
    entradas = torch.stack(entradas)

    resultados = []
    mapas_por_grupo = {}
    for (branch, seed), grupo in grupos:
        backbone, tipo_entrada, _ = BRANCH_FILES[branch]
        modelo = carregar_modelo(backbone, len(dataset.class_names), tipo_entrada, caminhos[branch, seed])
        if modelo is None:
            raise FileNotFoundError(caminhos[branch, seed])
        modelo = modelo.to(device).eval()
        atuais, mapas_brutos = _inferir_e_gerar_mapas(grupo, modelo, entradas, device)
        comparacoes, mapas = comparar_com_prototipos(atuais, mapas_brutos, pacientes, ssim_entre_mapas)
        resultados.append(comparacoes)
        mapas_por_grupo[branch, seed] = mapas
        del modelo, mapas_brutos

    detalhes = pd.concat(resultados, ignore_index=True)
    por_seed, correlacoes = resumir_correlacoes(detalhes)
    conferencia = (
        detalhes.assign(abs_margin_delta=(detalhes["margin"] - detalhes["margin_csv"]).abs())
        .groupby(["dataset", "branch", "seed"], as_index=False)
        .agg(
            n_observations=("sample", "size"),
            n_classes_changed=("class_changed", "sum"),
            n_top2_outside_tolerance=("top2_outside_tolerance", "sum"),
            max_abs_margin_delta=("abs_margin_delta", "max"),
        )
    )
    tabelas = {
        "contagens_predicoes.csv": resumir_grupos(predicoes),
        "conferencia_predicoes.csv": conferencia,
        "resultados_detalhados.csv": detalhes,
        "resumo_grupos.csv": resumir_grupos(detalhes),
        "correlacoes_por_seed.csv": por_seed,
        "resumo_correlacoes.csv": correlacoes,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exemplos = _selecionar_exemplos(detalhes, mapas_por_grupo, pacientes)
    with TemporaryDirectory(dir=output_dir) as temporario:
        destino = Path(temporario)
        for nome, tabela in tabelas.items():
            tabela.to_csv(destino / nome, index=False)
        salvar_figuras(detalhes, entradas, exemplos, dataset.class_names, CORTE_PRINCIPAL, destino)
        for arquivo in destino.iterdir():
            arquivo.replace(output_dir / arquivo.name)
    return detalhes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predicoes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--classes", nargs="+", help="Classes na mesma ordem de predicoes.csv.")
    parser.add_argument("--device")
    parser.add_argument("--pacientes-csv", type=Path, help="CSV opcional com sample e patient_id.")
    parser.add_argument("--somente-predicoes", action="store_true", help="Gera apenas as contagens.")
    args = parser.parse_args()
    if args.somente_predicoes:
        dados = preparar_predicoes(pd.read_csv(args.predicoes), args.dataset)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output_dir) as temporario:
            caminho = Path(temporario) / "contagens_predicoes.csv"
            resumir_grupos(dados).to_csv(caminho, index=False)
            caminho.replace(args.output_dir / caminho.name)
        return
    if not (args.test_dir and args.models_dir and args.classes):
        parser.error("Informe --test-dir, --models-dir e --classes para calcular os mapas.")

    from model.dataset import EnsembleTestDataset
    from model.utils import DEVICE, transform_originais, transform_recplot

    dataset = EnsembleTestDataset(args.test_dir, args.classes, transform_originais, transform_recplot)
    executar_limiar_v2(
        args.predicoes, dataset, args.dataset, args.output_dir, args.models_dir,
        device=args.device or DEVICE, pacientes_csv=args.pacientes_csv,
    )


if __name__ == "__main__":
    main()
