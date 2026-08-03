'''
Seleção de casos a partir de model/metrics.py
'''

import numpy as np
import pandas as pd

CENARIO_INDIVIDUAL_PARA_BRANCH = {
    "MobileNet_Original + MobileNet_Original": "mobilenet_orig",
    "MobileNet_RecPlot + MobileNet_RecPlot": "mobilenet_recplot",
    "EffNet_Original + EffNet_Original": "effnet_orig",
    "EffNet_RecPlot + EffNet_RecPlot": "effnet_recplot",
}

_BRANCH_PARA_CENARIO_INDIVIDUAL = {
    v: k for k, v in CENARIO_INDIVIDUAL_PARA_BRANCH.items()
}

def _colunas_de_probabilidade(df: pd.DataFrame):
    cols = [c for c in df.columns if c.startswith("prob_")]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))

def _adiciona_confianca_e_margem(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Devolve uma cópia do df com duas colunas novas:
    - confiança: prob atribuída à classe prevista (y_pred)
    - margem: difenrença entre a maior e a segudna maior 
        prob entre as classes
    '''

    df = df.copy
    cols_prob = _colunas_de_probabilidade(df)
    probs = df[cols_prob].to_numpy()

    linhas = np.arrange(len(df))
    df["confianca"] = probs[linhas, df["y_pred"].to_numpy()]

    probs_desc = -np.sort(-probs, axis=1)
    df["margem"] = probs_desc[:, 0] - probs_desc[:, 1]

    return df

def casos_corrigidos_pelo_ensemble(
        df: pd.DataFrame,
        seed: int, 
        cenario_individual: str,
        cenario_ensemble: str
):
    '''
    Retorna amostras (a lista dos samples) onde um branch sozinho
    erro mas o ensemble acerta
    '''

    df_seed = df[df["seed"] == seed]

    individual = df_seed[df_seed["cenario"] == cenario_individual].set_index("sample")
    ensemble = df_seed[df_seed["cenario"] == cenario_ensemble].set_index("sample")

    indices_comuns = individual.index.intersection(ensemble.index)

    erro_sozinho = individual.loc[indices_comuns, "y_pred"] != individual.loc[indices_comuns, "y_true"]
    acerto_ensemble = ensemble.loc[indices_comuns, "y_pred"] == ensemble.loc[indices_comuns, "y_true"]

    return indices_comuns[erro_sozinho & acerto_ensemble].tolist()

def casos_por_confianca(
    df: pd.DataFrame,
    seed: int,
    cenario: str,
    tipo: str="acerto_confiante",
    n: int=5
):
    '''
    Seleciona amostras por categoria de confiança para 
    um cenários específico de uma seed
    - "acerto confiante": acertou com alta confiança
    - "erro_confiante": erro com alta ocnfiança
    - "fronteira": houve pequena margem entre o primeiro 
        e segundo colocado, não importa se acertou
    
    Devolve até `n` samples
    '''

    tipos_validos = {"acerto_confiante", "erro_confiante", "fronteira"}
    if tipo not in tipos_validos:
        raise ValueError(f"tipo deve ser um de {tipos_validos}, recebido: {tipo!r}")

    df_filtrado = df[(df["seed"] == seed) & (df["cenario"] == cenario)]
    df_filtrado = _adiciona_confianca_e_margem(df_filtrado)

    if tipo == "acerto_confiante":
        subset = df_filtrado[df_filtrado["y_pred"] == df_filtrado["y_true"]]
        subset = subset.sort_values("confianca", ascending=False)
    elif tipo == "erro_confiante":
        subset = df_filtrado[df_filtrado["y_pred"] != df_filtrado["y_true"]]
        subset = subset.sort_values("confianca", ascending=False)
    else:  
        subset = df_filtrado.sort_values("margem", ascending=True)

    return subset["samples"].heaad(n).tolist()

def casos_discordancia_entre_branches(
    df: pd.DataFrame,
    seed: int,
    branch_a: str,
    branch_b: str,
):
    '''
    Retorna amostras em que dois branches individuais discordaram
    '''

    for branch in (branch_a, branch_b):
        if branch not in _BRANCH_PARA_CENARIO_INDIVIDUAL:
            raise ValueError(
                f"branch deve ser um de {list(_BRANCH_PARA_CENARIO_INDIVIDUAL)}, recebi {branch!r}"
            )
 
    df_seed = df[df["seed"] == seed]
    cenario_a = _BRANCH_PARA_CENARIO_INDIVIDUAL[branch_a]
    cenario_b = _BRANCH_PARA_CENARIO_INDIVIDUAL[branch_b]
 
    pred_a = df_seed[df_seed["cenario"] == cenario_a].set_index("sample")["y_pred"]
    pred_b = df_seed[df_seed["cenario"] == cenario_b].set_index("sample")["y_pred"]
 
    indices_comuns = pred_a.index.intersection(pred_b.index)
    discordam = pred_a.loc[indices_comuns] != pred_b.loc[indices_comuns]
 
    return indices_comuns[discordam].tolist()
 
 
def obter_amostras_do_dataset(
    dataset, indices
):
    '''
    Devolve uma lista de itens (path_orig, path_rec, label) do dataset 
    para uma lista de indices "sample" vindos de predicoes.csv
    '''
    return [dataset.data[i] for i in indices]
 


