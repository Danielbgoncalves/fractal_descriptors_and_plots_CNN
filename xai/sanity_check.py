"""
Sanity check por randomização progressiva de parâmetros.

O procedimento compara, para cada imagem, a atribuição do modelo treinado
com as atribuições obtidas após randomizar progressivamente os blocos da
rede, da saída para a entrada.

A seleção das imagens é estratificada apenas pelo rótulo verdadeiro. A mesma
lista de amostras é usada em todas as seeds, branches e métodos, permitindo
comparações pareadas. A classe explicada é a classe predita pelo modelo
treinado, mantida fixa durante toda a cascata.
"""

from __future__ import annotations

import copy
import os
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim

from xai.attributions import gerar_atribuicao
from xai.loader import BRANCH_FILES, carregar_todos_os_branches, obter_camada_alvo


BRANCH_TO_SCENARIO = {
    "mobilenet_orig": "MobileNet_Original + MobileNet_Original",
    "mobilenet_recplot": "MobileNet_RecPlot + MobileNet_RecPlot",
    "effnet_orig": "EffNet_Original + EffNet_Original",
    "effnet_recplot": "EffNet_RecPlot + EffNet_RecPlot",
}


def _reinicializar_bloco(modulo: torch.nn.Module) -> None:
    """Reinicializa os parâmetros de todos os módulos do bloco."""
    for submodulo in modulo.modules():
        reset = getattr(submodulo, "reset_parameters", None)
        if callable(reset):
            reset()


def _blocos_randomizacao(modelo: torch.nn.Module):
    """
    Ordem da saída para a entrada: classificador, features[-1], ..., features[0].
    O último estágio corresponde à rede integralmente randomizada.
    """
    if not hasattr(modelo, "classifier") or not hasattr(modelo, "features"):
        raise TypeError("O modelo precisa expor os módulos 'classifier' e 'features'.")

    blocos = [("classifier", modelo.classifier)]
    for indice in reversed(range(len(modelo.features))):
        blocos.append((f"features[{indice}]", modelo.features[indice]))
    return blocos


def _para_heatmap_2d(attr: torch.Tensor) -> np.ndarray:
    """
    Converte atribuições (1,C,H,W) em mapa 2D de magnitude e normaliza
    individualmente para [0,1]. Para IG, a magnitude é usada, sem preservar
    o sinal das atribuições.
    """
    mapa = attr.detach().squeeze(0)

    if mapa.ndim == 3:
        mapa = mapa.abs().sum(dim=0)
    elif mapa.ndim != 2:
        raise ValueError(f"Formato de atribuição não suportado: {tuple(attr.shape)}")

    mapa = mapa.float().cpu().numpy()
    mapa = np.nan_to_num(mapa, nan=0.0, posinf=0.0, neginf=0.0)

    minimo = float(mapa.min())
    maximo = float(mapa.max())
    amplitude = maximo - minimo

    if amplitude < 1e-12:
        return np.zeros_like(mapa, dtype=np.float32)

    return ((mapa - minimo) / amplitude).astype(np.float32)


def _calcular_similaridades(
    mapa_referencia: np.ndarray,
    mapa_atual: np.ndarray,
) -> tuple[float, float]:
    """
    Retorna Spearman e SSIM. Se algum mapa for constante, as métricas são
    indefinidas e retornam NaN, em vez de transformar ausência de variação
    em similaridade zero ou perfeita.
    """
    eps = 1e-12

    if (
        np.std(mapa_referencia) < eps
        or np.std(mapa_atual) < eps
    ):
        return np.nan, np.nan

    rho = spearmanr(
        mapa_referencia.ravel(),
        mapa_atual.ravel(),
    ).statistic

    if not np.isfinite(rho):
        rho = np.nan

    valor_ssim = ssim(
        mapa_referencia,
        mapa_atual,
        data_range=1.0,
    )

    return float(rho), float(valor_ssim)


def teste_rand_em_cascata(
    modelo_treinado: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    metodo: str = "gradcam",
    obter_camada=obter_camada_alvo,
    metadata: Optional[dict] = None,
    random_state: int = 0,
) -> List[dict]:
    """
    Executa uma cascata de randomização para uma imagem e um método XAI.

    A classe-alvo é fixa: deve ser a predição do modelo treinado. Isso evita
    que a classe explicada mude artificialmente entre os estágios.
    """
    if metodo not in {"gradcam", "ig", "occlusion"}:
        raise ValueError(f"Método não suportado: {metodo}")

    metadata = dict(metadata or {})
    modelo_treinado.eval()

    device = next(modelo_treinado.parameters()).device
    entrada = input_tensor.to(device)

    if entrada.ndim == 3:
        entrada = entrada.unsqueeze(0)

    def atribuicao(modelo: torch.nn.Module) -> np.ndarray:
        camada = obter_camada(modelo) if metodo == "gradcam" else None
        attr = gerar_atribuicao(
            modelo,
            entrada,
            int(target_class),
            camada_alvo=camada,
            metodo=metodo,
        )
        return _para_heatmap_2d(attr)

    mapa_referencia = atribuicao(modelo_treinado)
    desvio_referencia = float(np.std(mapa_referencia))

    rho_ref, ssim_ref = _calcular_similaridades(
        mapa_referencia,
        mapa_referencia,
    )

    resultados = [{
        **metadata,
        "method": metodo,
        "target_class": int(target_class),
        "step": 0,
        "block": "trained_model",
        "progress": 0.0,
        "spearman": 1.0 if np.isfinite(rho_ref) else np.nan,
        "ssim": 1.0 if np.isfinite(ssim_ref) else np.nan,
        "reference_map_std": desvio_referencia,
        "current_map_std": desvio_referencia,
        "reference_map_flat": bool(desvio_referencia < 1e-12),
        "current_map_flat": bool(desvio_referencia < 1e-12),
    }]

    # Isola o estado aleatório para que cada cascata seja reproduzível
    # sem alterar a sequência aleatória usada pelo restante do processo.
    dispositivos_cuda = []
    if device.type == "cuda":
        dispositivos_cuda = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]

    with torch.random.fork_rng(devices=dispositivos_cuda):
        torch.manual_seed(int(random_state))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(random_state))

        modelo_randomizado = copy.deepcopy(modelo_treinado)
        modelo_randomizado.eval()
        blocos = _blocos_randomizacao(modelo_randomizado)
        total_blocos = len(blocos)

        for step, (nome_bloco, modulo) in enumerate(blocos, start=1):
            _reinicializar_bloco(modulo)
            modelo_randomizado.eval()

            mapa_atual = atribuicao(modelo_randomizado)
            desvio_atual = float(np.std(mapa_atual))
            rho, valor_ssim = _calcular_similaridades(
                mapa_referencia,
                mapa_atual,
            )

            resultados.append({
                **metadata,
                "method": metodo,
                "target_class": int(target_class),
                "step": step,
                "block": nome_bloco,
                "progress": step / total_blocos,
                "spearman": rho,
                "ssim": valor_ssim,
                "reference_map_std": desvio_referencia,
                "current_map_std": desvio_atual,
                "reference_map_flat": bool(desvio_referencia < 1e-12),
                "current_map_flat": bool(desvio_atual < 1e-12),
            })

    return resultados


def selecionar_amostras_estratificadas(
    dataset,
    n_por_classe: int = 5,
    random_state: int = 2026,
) -> List[int]:
    """
    Seleciona uma amostra aleatória reprodutível por classe verdadeira.
    A seleção não depende das predições dos modelos.
    """
    if n_por_classe < 1:
        raise ValueError("n_por_classe deve ser maior ou igual a 1.")

    if not hasattr(dataset, "data"):
        raise AttributeError(
            "O dataset precisa expor o atributo 'data' para selecionar "
            "as classes sem carregar todas as imagens."
        )

    rotulos = []
    for item in dataset.data:
        if isinstance(item, dict):
            rotulos.append(int(item["label"]))
        else:
            rotulos.append(int(item[1]))

    rotulos = np.asarray(rotulos)
    rng = np.random.default_rng(random_state)

    selecionadas = []
    for classe in np.unique(rotulos):
        indices = np.flatnonzero(rotulos == classe)
        quantidade = min(n_por_classe, len(indices))
        escolhidas = rng.choice(indices, size=quantidade, replace=False)
        selecionadas.extend(int(i) for i in escolhidas)

    return sorted(selecionadas)


def _indice_predicoes(
    df_predicoes: pd.DataFrame,
    seed: int,
    branch: str,
) -> Dict[int, dict]:
    """Indexa as predições individuais de um branch por índice de amostra."""
    if branch not in BRANCH_TO_SCENARIO:
        raise KeyError(f"Branch sem cenário individual mapeado: {branch}")

    cenario = BRANCH_TO_SCENARIO[branch]
    recorte = df_predicoes[
        (df_predicoes["seed"].astype(int) == int(seed))
        & (df_predicoes["cenario"] == cenario)
    ]

    if recorte.empty:
        raise ValueError(
            f"Não há predições para seed={seed}, branch={branch}, "
            f"cenario='{cenario}'."
        )

    if recorte["sample"].duplicated().any():
        raise ValueError(
            f"Há índices de amostra duplicados em seed={seed}, branch={branch}."
        )

    return {
        int(row["sample"]): row.to_dict()
        for _, row in recorte.iterrows()
    }


def _resumir_curvas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Primeiro calcula a média entre imagens dentro de cada seed; depois
    resume as oito seeds. Assim, o desvio entre seeds não trata cada
    imagem-seed como uma observação independente.
    """
    chaves_seed = [
        "dataset", "method", "branch", "seed",
        "step", "block", "progress",
    ]

    por_seed = (
        df.groupby(chaves_seed, dropna=False)
        .agg(
            spearman_seed=("spearman", "mean"),
            ssim_seed=("ssim", "mean"),
            n_images=("sample", "nunique"),
            n_spearman_valid=("spearman", "count"),
            n_ssim_valid=("ssim", "count"),
        )
        .reset_index()
    )

    chaves = [
        "dataset", "method", "branch",
        "step", "block", "progress",
    ]

    resumo = (
        por_seed.groupby(chaves, dropna=False)
        .agg(
            spearman_mean=("spearman_seed", "mean"),
            spearman_std=("spearman_seed", "std"),
            spearman_median=("spearman_seed", "median"),
            ssim_mean=("ssim_seed", "mean"),
            ssim_std=("ssim_seed", "std"),
            ssim_median=("ssim_seed", "median"),
            n_seeds=("seed", "nunique"),
            n_images_mean=("n_images", "mean"),
            n_spearman_valid=("n_spearman_valid", "sum"),
            n_ssim_valid=("n_ssim_valid", "sum"),
        )
        .reset_index()
    )

    return resumo


def _calcular_auc(df: pd.DataFrame):
    """Calcula AUC da curva de similaridade por imagem e seed."""
    chaves = ["dataset", "method", "branch", "seed", "sample"]
    linhas = []

    for valores, grupo in df.groupby(chaves):
        grupo = grupo.sort_values("progress")
        linha = dict(zip(chaves, valores))

        for metrica in ("spearman", "ssim"):
            validos = grupo[["progress", metrica]].dropna()
            if len(validos) >= 2:
                linha[f"auc_{metrica}"] = float(np.trapz(
                    validos[metrica].to_numpy(),
                    validos["progress"].to_numpy(),
                ))
            else:
                linha[f"auc_{metrica}"] = np.nan

        linhas.append(linha)

    auc_amostra = pd.DataFrame(linhas)

    por_seed = (
        auc_amostra.groupby(
            ["dataset", "method", "branch", "seed"],
            dropna=False,
        )
        .agg(
            auc_spearman=("auc_spearman", "mean"),
            auc_ssim=("auc_ssim", "mean"),
            n_samples=("sample", "nunique"),
        )
        .reset_index()
    )

    resumo = (
        por_seed.groupby(
            ["dataset", "method", "branch"],
            dropna=False,
        )
        .agg(
            auc_spearman_mean=("auc_spearman", "mean"),
            auc_spearman_std=("auc_spearman", "std"),
            auc_ssim_mean=("auc_ssim", "mean"),
            auc_ssim_std=("auc_ssim", "std"),
            n_seeds=("seed", "nunique"),
            n_samples_mean=("n_samples", "mean"),
        )
        .reset_index()
    )

    return auc_amostra, por_seed, resumo


def _plotar_curvas(
    resumo: pd.DataFrame,
    dataset_nome: str,
    metodo: str,
    metrica: str,
    output_path: str,
) -> Figure:
    coluna_media = f"{metrica}_mean"
    coluna_std = f"{metrica}_std"

    fig, ax = plt.subplots(figsize=(10, 6))
    recorte = resumo[
        (resumo["dataset"] == dataset_nome)
        & (resumo["method"] == metodo)
    ]

    for branch, grupo in recorte.groupby("branch"):
        grupo = grupo.sort_values("progress")
        x = grupo["progress"].to_numpy(dtype=float)
        y = grupo[coluna_media].to_numpy(dtype=float)
        desvio = grupo[coluna_std].fillna(0).to_numpy(dtype=float)

        ax.plot(x, y, marker="o", linewidth=1.8, label=branch)
        ax.fill_between(
            x,
            y - desvio,
            y + desvio,
            alpha=0.15,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("Proporção de blocos randomizados")
    ax.set_ylabel(f"Similaridade ({metrica})")
    ax.set_title(f"{dataset_nome} — sanity check — {metodo} — {metrica}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return fig


def executar_sanity_check_dataset(
    dataset,
    df_predicoes: pd.DataFrame,
    dataset_nome: str,
    class_names: List[str],
    seeds: Iterable[int],
    models_dir: str,
    output_dir: str,
    metodos: Iterable[str] = ("gradcam", "ig", "occlusion"),
    n_por_classe: int = 5,
    random_state: int = 2026,
    device: str = "cpu",
):
    """
    Executa o sanity check em todos os métodos, seeds e quatro branches.

    Arquivos gerados:
      - resultados_detalhados.csv: cada amostra em cada estágio da cascata;
      - resumo_curvas.csv: média entre imagens por seed e resumo entre seeds;
      - resumo_por_classe.csv e resumo_por_correcao.csv;
      - auc_por_amostra.csv, auc_por_seed.csv e resumo_auc.csv;
      - amostras_selecionadas.csv e curvas de Spearman/SSIM.
    """
    metodos = list(metodos)
    seeds = list(seeds)

    os.makedirs(output_dir, exist_ok=True)

    amostras = selecionar_amostras_estratificadas(
        dataset,
        n_por_classe=n_por_classe,
        random_state=random_state,
    )

    if not amostras:
        raise ValueError("Nenhuma amostra foi selecionada.")

    # Carrega as imagens uma única vez e reutiliza os mesmos índices
    # para todas as seeds, arquiteturas, representações e métodos.
    cache = {indice: dataset[indice] for indice in amostras}

    linhas_amostras = []
    for indice in amostras:
        _, _, label = cache[indice]
        classe = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
        linhas_amostras.append({
            "sample": indice,
            "y_true": classe,
            "true_class_name": class_names[classe],
        })

    pd.DataFrame(linhas_amostras).to_csv(
        os.path.join(output_dir, "amostras_selecionadas.csv"),
        index=False,
    )

    resultados = []

    for seed in seeds:
        print(f"[SANITY] {dataset_nome} — carregando seed {seed}")

        branches = carregar_todos_os_branches(
            seed=seed,
            num_classes=len(class_names),
            models_dir=models_dir,
            device=device,
        )

        for branch, modelo in branches.items():
            if modelo is None:
                print(f"[SANITY][AVISO] branch {branch} indisponível na seed {seed}")
                continue

            predicoes = _indice_predicoes(df_predicoes, seed, branch)
            tipo_entrada = BRANCH_FILES[branch][1]

            for indice in amostras:
                img_orig, img_rec, label = cache[indice]
                y_true = int(label.item()) if isinstance(label, torch.Tensor) else int(label)

                if indice not in predicoes:
                    raise ValueError(
                        f"Predição ausente: seed={seed}, branch={branch}, sample={indice}"
                    )

                pred = predicoes[indice]
                y_pred = int(pred["y_pred"])

                if int(pred["y_true"]) != y_true:
                    raise ValueError(
                        "Divergência entre a classe do dataset e predicoes.csv "
                        f"(seed={seed}, branch={branch}, sample={indice}). "
                        "Verifique se a ordenação do dataset de teste é a mesma "
                        "usada para gerar o CSV."
                    )

                entrada = img_orig if tipo_entrada == "originais" else img_rec
                probabilidade = float(pred.get(f"prob_{y_pred}", np.nan))

                metadata = {
                    "dataset": dataset_nome,
                    "branch": branch,
                    "seed": int(seed),
                    "sample": int(indice),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "true_class_name": class_names[y_true],
                    "predicted_class_name": class_names[y_pred],
                    "correct": bool(y_true == y_pred),
                    "predicted_class_probability": probabilidade,
                }

                for metodo in metodos:
                    print(
                        f"[SANITY] {dataset_nome} | seed={seed} | "
                        f"{branch} | sample={indice} | {metodo}"
                    )

                    resultados.extend(
                        teste_rand_em_cascata(
                            modelo_treinado=modelo,
                            input_tensor=entrada,
                            target_class=y_pred,
                            metodo=metodo,
                            metadata=metadata,
                            random_state=(
                                random_state
                                + int(seed) * 1_000_003
                                + int(indice) * 10_007
                                + list(BRANCH_FILES).index(branch) * 101
                                + list(metodos).index(metodo)
                            ) % (2**31 - 1),
                        )
                    )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not resultados:
        raise RuntimeError("O sanity check não produziu resultados.")

    df = pd.DataFrame(resultados)
    df.to_csv(
        os.path.join(output_dir, "resultados_detalhados.csv"),
        index=False,
    )

    resumo = _resumir_curvas(df)
    resumo.to_csv(
        os.path.join(output_dir, "resumo_curvas.csv"),
        index=False,
    )

    # Análises complementares: diferenças entre classes e entre acertos/erros.
    for coluna, nome_arquivo in (
        ("true_class_name", "resumo_por_classe.csv"),
        ("correct", "resumo_por_correcao.csv"),
    ):
        agrupamento = (
            df.groupby(
                ["dataset", "method", "branch", coluna, "step", "block", "progress"],
                dropna=False,
            )
            .agg(
                spearman_mean=("spearman", "mean"),
                spearman_std=("spearman", "std"),
                ssim_mean=("ssim", "mean"),
                ssim_std=("ssim", "std"),
                n_images=("sample", "nunique"),
                n_observations=("sample", "count"),
            )
            .reset_index()
        )
        agrupamento.to_csv(
            os.path.join(output_dir, nome_arquivo),
            index=False,
        )

    auc_amostra, auc_seed, resumo_auc = _calcular_auc(df)
    auc_amostra.to_csv(
        os.path.join(output_dir, "auc_por_amostra.csv"),
        index=False,
    )
    auc_seed.to_csv(
        os.path.join(output_dir, "auc_por_seed.csv"),
        index=False,
    )
    resumo_auc.to_csv(
        os.path.join(output_dir, "resumo_auc.csv"),
        index=False,
    )

    for metodo in metodos:
        for metrica in ("spearman", "ssim"):
            _plotar_curvas(
                resumo=resumo,
                dataset_nome=dataset_nome,
                metodo=metodo,
                metrica=metrica,
                output_path=os.path.join(
                    output_dir,
                    f"curva_{metrica}_{metodo}.png",
                ),
            )

    configuracao = pd.DataFrame([{
        "dataset": dataset_nome,
        "seeds": ",".join(str(seed) for seed in seeds),
        "methods": ",".join(metodos),
        "n_per_class": n_por_classe,
        "n_samples_selected": len(amostras),
        "random_state_selection": random_state,
        "target_class_rule": "classe predita pelo modelo treinado, fixa na cascata",
        "randomization_order": "classifier -> features[-1] -> ... -> features[0]",
        "map_rule": "soma das magnitudes dos canais, normalização min-max por mapa",
    }])
    configuracao.to_csv(
        os.path.join(output_dir, "configuracao.csv"),
        index=False,
    )

    return df, resumo, resumo_auc
