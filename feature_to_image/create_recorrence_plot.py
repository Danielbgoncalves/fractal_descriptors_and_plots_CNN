# Referência no trabalho de  Hui Yang
# Affiliation: 
    # The Pennsylvania State University
    # 310 Leohard Building, University Park, PA
    # Email: yanghui@gmail.com
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def create_recorrence_plot(signal):
    N = len(signal)
    buffer = np.zeros((N,N))

    for i in range(N):
        x0 = i
        for j in range(i, N):
            y0 = j
            distance = np.linalg.norm(signal[i,:] - signal[j,:])
            buffer[x0, y0] = distance
            buffer[y0, x0] = distance


    return buffer

