import numpy as np
from numba import njit

'''
Função auxiliar usadas nos arquivos MATLAB mas sem 
correspondência direta em função do python (aparentemente)
Aí implementei
'''


@njit
def mat2gray(arr):
    arr = arr.astype(np.float64)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val: # dividir por zero costuma dar mal
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)