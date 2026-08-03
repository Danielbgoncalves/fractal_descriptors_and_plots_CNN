'''
Gera métricas de qualiade das explicações (Quantus) por seed x branches
3 métricas sobrindo 3 perguntas:
    - faithfulnes: a região marcada como importante é, de fato, 
      a mais usada pelo modelo na predição ? 
    - robustness: pertubar a entrada levemente muda muito a avaliação ?
    - complexity: a explicação está concentrada em uma área pequena
      ou espalhada ?
'''
import os
import numpy as np
import pandas as pd
import quantus
import torch
import torch.nn as nn
 
from xai.attributions import gerar_atribuicao
from xai.loader import BRANCH_FILES, carregar_todos_os_branches, obter_camada_alvo

def _criar_explain_func(metodo: str="gradcam"):
    '''
    Essa função é usada pelo Quantus no cálculo de robustness,
    ele exige uma função que retorne a outra porque ele pertuba 
    a entrada a cada iteração
    '''

    def _explain(model: nn.Module, inputs: np.ndarray, targets: np.ndarray, **kwards):
        device = next(model.parameters()).device
        x = torch.tensor(inputs, dtype=torch.float32, device=device)
        camada = obter_camada_alvo(model) if metodo == "gradcam" else None

        saida = []
        for i in range(x.shape[0]):
            a = gerar_atribuicao(model, x[i : i + 1], int(targets[i]), camada_alvo=camada, metodo=metodo)

            saida.append(a.detach().cpu().numpy())
        return np.concatenate(saida, axis=0)

    return _explain

def calcula_matricas_xai(
        modelo: nn.Module,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        metodo: str="gradcam",
        nr_runs_faithfulness: int=32,
        subset_size_faithfulness: int=32,
        nr_samples_robustness: int=10
):
    '''
    Calcula as 3 métricas para um conjunto (branch, seed)
    x_batch: (N, C, H, W) já normalizado (a mesma entrada que vai pro
        modelo, não a imagem "crua"). y_batch: (N,) com as classes-alvo.
    '''
    device = str(next(modelo.parameters()).device)
    explain_func = _criar_explain_func(metodo)

    a_batch = explain_func(modelo, x_batch, y_batch)

    faithfulness = quantus.FaithfulnessCorrelation(
        nr_runs=nr_runs_faithfulness,
        subset_size=subset_size_faithfulness,
        return_aggregate=False,
        display_progressbar=False,
        disable_warnings=True
    )(model=modelo, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch,
      channel_first=True, device=device)

    robustness = quantus.MaxSensitivity(
        nr_samples=nr_samples_robustness,
        return_aggregate=False,
        display_progressbar=False,
        disable_warnings=True,
    )(model=modelo, x_batch=x_batch, y_batch=y_batch, explain_func=explain_func,
      channel_first=True, device=device)

    complexity = quantus.Sparseness(
        return_aggregate=False, 
        display_progressbar=False,
        disable_warnings=True
    )(model=modelo, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch,
      channel_first=True, device=device)

    return {
        "faithfulness": np.asarray(faithfulness, dtype=float),
        "robustness": np.asarray(robustness, dtype=float),
        "complexity": np.asarray(complexity, dtype=float),
    }

def xai_metrics_to_csv(
  seeds,
  num_classes: int,
  models_dir: str,
  dataset,
  output_dir: str,
  metodo: str="gradcam",
  n_amostras = None,
  device: str="cuda",
  nr_runs_faithfulness: int=32,
  subset_size_faithfulness: int=32,
  nr_samples_robustness: int=10
) -> pd.DataFrame:
    '''
    Roda calcula_metricas_xai() para as seeds x 4 branches e salva em CSV
    uma tabela por amostra (xai_metrics.csv) e agregado (xai_mestrics_mean.csv)
    dataset: do memso formato usando em mol/run.py
    n_amostras: quantas amostras do dataset usar, se =None usa todas
    '''

    n_disponivel = len(dataset)
    n_usar = n_disponivel if n_amostras is None else min(n_amostras, n_disponivel)

    imgs_orig, imgs_rec, labels, = [], [], []
    for i in range(n_usar):
        img_orig, img_rec, label = dataset[i]
        imgs_orig.append(img_orig)
        imgs_rec.append(img_rec)
        labels.append(int(label.item()) if isinstance(label, torch.Tensor) else int(label))

    x_orig = torch.stack(imgs_orig).numpy()        
    x_rec = torch.stack(imgs_rec).numpy()        
    y = np.array(labels)

    linhas = []

    for seed in seeds:
        print(f"[xai_metrics] seed {seed} ({n_usar} amostras, método={metodo})")
        branches = carregar_todos_os_branches(seed, num_classes, models_dir=models_dir, device=device)

        for nome_branch, modelo in branches.items():
            if modelo is None:
                print(f"    AVISO: pulando {nome_branch}, modelo não carregado")
                continue

            tipo_dataset = BRANCH_FILES[nome_branch][1] # "originais" ou "recplot"
            x_batch = x_orig if tipo_dataset == "originais" else x_rec

            metricas = calcula_matricas_xai(
                modelo, x_batch, y, metodo=metodo,
                nr_runs_faithfulness=nr_runs_faithfulness,
                subset_size_faithfulness=subset_size_faithfulness,
                nr_samples_robustness=nr_samples_robustness
            )

            for i_amostra in range(n_usar):
                for nome_metrica, valores in metricas.items():
                    linhas.append({
                        "seed": seed,
                        "branch": nome_branch,
                        "amostra": i_amostra,
                        "metrica": nome_metrica,
                        "valor": float(valores[i_amostra])
                    })

    df = pd.DataFrame(linhas)

    caminho_df = os.path.join(output_dir, "xai_metricas.csv")
    caminho_df_mean = os.path.join(output_dir, "xai_metricas_mean.csv")

    df.to_csv(caminho_df, index=False)

    df_mean = (
        df.groupby(["branch", "metrica"])["valor"]
        .agg(media="mean", std="std")
        .reset_index()
    )
    df_mean.to_csv(caminho_df_mean, index=False)

    return df

def mostrar_metricas_xai(df: pd.DataFrame) -> None:
    resumo = df.groupby(["branch", "metrica"])["valor"].agg(["mean", "std"])
    print(resumo)
