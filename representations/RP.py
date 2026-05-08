'''

'''
import numpy as np
# from create_recorrence_plot import create_recorrence_plot
import imageio.v3 as iio
from numba import njit
import os

@njit
def create_recorrence_plot(signal):
    N = signal.shape[0]
    buffer = np.zeros((N,N))

    for i in range(N):
        x0 = i
        for j in range(i, N):
            y0 = j
            distance = abs(signal[i, 0] - signal[j, 0])
            buffer[x0, y0] = distance
            buffer[y0, x0] = distance

    return buffer

@njit
def create_recorrence_plot_bin(signal, epsilon=0.15):
    """
    signal: array 2D de formato (N, 1)
    epsilon: limiar de distância. Valores comuns estão entre 0.1 e 0.2
             para sinais normalizados entre 0 e 1.
    """
    N = signal.shape[0]
    buffer = np.zeros((N, N))

    for i in range(N):
        # A diagonal principal sempre tem distância 0, logo é sempre 1 no RP
        buffer[i, i] = 1.0 
        
        for j in range(i + 1, N):
            # Distância absoluta para sinais 1D (muito mais rápido que linalg.norm)
            distance = abs(signal[i, 0] - signal[j, 0])
            
            # Função Heaviside: se a distância for menor que epsilon, é 1. Senão, 0.
            if distance <= epsilon:
                buffer[i, j] = 1.0
                buffer[j, i] = 1.0
            else:
                buffer[i, j] = 0.0
                buffer[j, i] = 0.0

    return buffer

def processar_features(n, new_features, tamanho_img):
    h = w = tamanho_img 
    imgs = np.zeros((n, h, w))

    for i in range(n):
        signal = new_features[i, :].reshape(-1, 1)
        # com "_bin" se de acordo com o artigo do zanchas e sem o "_bin" se de acordo com o artigo do Guilherme
        channel = create_recorrence_plot(signal) 
        imgs[i,:,:] = channel 
    
    return imgs

def normalizar_linhas_em_blocos(linha, tamanho_bloco=20):
    '''
    Normaliza os valores de uma única linha (imagem), bloco por bloco 
    (cada bloco = conjunto de descritores que fazem sentido juntos
    por exemplo os 20 MinkLAC ou os 20 Euclnn)
    '''

    linha_norm = np.zeros_like(linha)
    m = len(linha) # num de colunas

    for i in range(0, m, tamanho_bloco):
        fim = min(i + tamanho_bloco, m)
        bloco = linha[i:fim].astype(np.float64)
        min_val, max_val = np.min(bloco), np.max(bloco)

        if max_val > min_val:
            linha_norm[i:fim] = (bloco - min_val) / (max_val - min_val)
        else:
            linha_norm[i:fim] = 0.0
    
    return linha_norm
        

def generate_RP(features):
    '''
    ENTRADA: features: matriz numpy com descritores da PERCOLAÇÃO
    SAIDA: cada linha da matriz gera uma imagem 2D criada com reshape RecPlot
    '''
    n = features.shape[0] # quantidade de linhas (imagens a serem geradas)
    m = features.shape[1] # quantidade de colunas (descritores, imagem gerada é mxm)

    new_features = np.zeros((n, m))
    
    # Os descritores são 20 Minkp + 20 Minkg + 20 Minkh 
    # + o memos para Eucl e Manh
    # logo, sempre de 20 em 20 estão descritores relacionados
    for i in range(n):
        new_features[i, :] = normalizar_linhas_em_blocos(features[i, :], tamanho_bloco=20)
        # new_features[:, i:i+20] = mat2gray(features[:,  i:i+20])

    imgs = processar_features(n, new_features, m)

    return imgs