import numpy as np
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

def normalizar_features_globalmente(features, tamanho_bloco=20, mins_treino=None, maxs_treino=None):
    '''
    Normaliza a matriz de features inteira (todas as amostras simultaneamente),
    mas respeitando os "blocos" de descritores (ex: a cada 20 colunas).
    
    - Se mins_treino/maxs_treino forem None: Assume que é o conjunto de TREINO.
      Calcula e retorna os mins/maxs para serem usados depois.
    - Se forem passados: Assume que é Validação ou TESTE, e aplica os valores do treino.
    '''
    n = features.shape[0]
    m = features.shape[1]
    new_features = np.zeros((n, m))
    
    # Flag para saber se estamos no Treino (calculando) ou no Teste (aplicando)
    calculando_treino = (mins_treino is None) or (maxs_treino is None)
    
    mins_salvos = []
    maxs_salvos = []
    
    for idx_bloco, i in enumerate(range(0, m, tamanho_bloco)):
        fim = min(i + tamanho_bloco, m)
        bloco = features[:, i:fim].astype(np.float64) # Pega TODAS as linhas, e colunas do bloco
        
        if calculando_treino:
            # Acha o min/max do bloco em todo o dataset de TREINO
            min_val = np.min(bloco)
            max_val = np.max(bloco)
            mins_salvos.append(min_val)
            maxs_salvos.append(max_val)
        else:
            # Usa o min/max que a rede aprendeu no treino
            min_val = mins_treino[idx_bloco]
            max_val = maxs_treino[idx_bloco]

        # Evita divisão por zero
        if max_val > min_val:
            # Normalização Min-Max padrão
            bloco_norm = (bloco - min_val) / (max_val - min_val)
            
            # Se for teste, alguns valores podem estourar os limites [0, 1]. Clipamos para garantir:
            if not calculando_treino:
                bloco_norm = np.clip(bloco_norm, 0.0, 1.0)
                
            new_features[:, i:fim] = bloco_norm
        else:
            new_features[:, i:fim] = 0.0
            
    if calculando_treino:
        return new_features, mins_salvos, maxs_salvos
    else:
        return new_features


def generate_RP_treino(features_treino):
    ''' Gera as imagens de Treino e guarda os parâmetros de normalização '''
    n = features_treino.shape[0]
    m = features_treino.shape[1]
    
    new_features, mins, maxs = normalizar_features_globalmente(features_treino, tamanho_bloco=20)
    
    imgs = processar_features(n, new_features, m)
    
    return imgs, mins, maxs
    #return imgs


def generate_RP_teste(features_teste, mins_treino, maxs_treino):
    ''' Gera as imagens de Teste/Validação usando a escala do Treino '''
    n = features_teste.shape[0]
    m = features_teste.shape[1]
    
    new_features = normalizar_features_globalmente(
        features_teste, 
        tamanho_bloco=20, 
        mins_treino=mins_treino, 
        maxs_treino=maxs_treino
    )
    
    imgs = processar_features(n, new_features, m)
    return imgs