'''
Converte os atributos locais salvos no arquivo CSV com uso do script
SaveCSVPercLACDF3Distances em uma imagem de atributos fractais com
uso da técnica recurrence plot

input: 
    diretório de destino das imagens geradas;
    actual dir precisa n
    cvs lido como matriz;
'''
import numpy as np
from create_recorrence_plot import create_recorrence_plot
import imageio.v3 as iio
from numba import njit
import os

@njit
def mat2gray(arr):
    arr = arr.astype(np.float64)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val == min_val: # dividir por zero costuma dar mal
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

def processar_features(n, new_features):
    h = w = 100 # aqui deveria ser h = w = len(signal) mas já sabemos q vai ser 100
    imgs = np.zeros((n, h, w, 3))

    for i in range(n):
        signal_r = new_features[i, :, 0].reshape(-1, 1)
        signal_g = new_features[i, :, 1].reshape(-1, 1)
        signal_b = new_features[i, :, 2].reshape(-1, 1)
    
        r_channel = create_recorrence_plot(signal_r)
        g_channel = create_recorrence_plot(signal_g)
        b_channel = create_recorrence_plot(signal_b)

        imgs[i,:,:,0] = r_channel 
        imgs[i,:,:,1] = g_channel 
        imgs[i,:,:,2] = b_channel
        imgs[i,:,:,:] = np.clip(imgs[i,:,:,:], 0, 1)
    
    return imgs

def reshapeRecPlot(destino, features, out_paths=None):
    '''
    destino: Onde salvar as imagens criadas (usado quando out_paths é None)
    features: matriz numpy resultado da leitura do df
    out_paths: lista opcional de caminhos completos para salvar cada imagem
    '''
    n = features.shape[0] # quantidade de linhas (imagens a serem geradas)
    featuresSplit = np.zeros((n, 100, 3))

    featuresSplit[:,:,0] = features[:,   0:100]
    featuresSplit[:,:,1] = features[:, 100:200]
    featuresSplit[:,:,2] = features[:, 200:300]

    new_features = np.zeros((n, 100, 3))
    
    new_features[:, 0:20 ,:] = mat2gray(featuresSplit[:,  0:20,:])
    new_features[:, 20:40 ,:] = mat2gray(featuresSplit[:, 20:40, :])
    new_features[:, 40:60 ,:] = mat2gray(featuresSplit[:, 40:60, :])
    new_features[:, 60:80 ,:] = mat2gray(featuresSplit[:, 60:80, :])
    new_features[:, 80:100 ,:] = mat2gray(featuresSplit[:,80:100, :])

    imgs = processar_features(n, new_features)

    if out_paths is None:
        path = f'{destino}/F-RecPlot'
        os.makedirs(path, exist_ok=True)
        for idx, img in enumerate(imgs):
            img_uint8 = (img * 255).astype(np.uint8)
            iio.imwrite(f'{path}/F-RecPlot{idx+1}.png', img_uint8)
    else:
        # out_paths deve conter caminhos completos (com nome de arquivo)
        for idx, img in enumerate(imgs):
            img_uint8 = (img * 255).astype(np.uint8)
            out_path = out_paths[idx]
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            iio.imwrite(out_path, img_uint8)
