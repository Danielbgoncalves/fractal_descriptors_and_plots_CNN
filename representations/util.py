import numpy as np
from numba import njit

@njit
def mat2gray(arr):
    arr = arr.astype(np.float64)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val: # dividir por zero costuma dar mal
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

def organize_by_nature(features):
    # Separa os descritores fractais por sua natureza fractal
    # não pela forma como a distância foi medida

    # PERCOLAÇÃO: 60 de Mink, 60 de Eucl, 60 de Manh = 180 pontos
    perc = np.hstack((features[:, 0:60], features[:, 100:160], features[:, 200:260]))

    # LACUNARIDADE: 20 de Mink, 20 de Eucl, 20 de Manh = 60 pontos
    lac = np.hstack((features[:, 60:80 ], features[:, 160:180 ], features[:, 260:280 ]))

    # GLOBAIS + DF: Os 63 finais + os 60 de "ante-passo da DF" (massa) = 123 pontos
    massa = np.hstack((features[:, 80:100], features[:, 180:200], features[:, 280:300]))
    globais = features[:, 300:363]
    global_df = np.hstack((massa, globais))

    return perc, lac, global_df    