import numpy as np
import imageio.v3 as iio
import os

from utils import mat2gray

def reshapeClassical(destino, features, out_paths=None):
    '''
    destino: Onde salvar as imagens criadas
    features: matriz numpy resultado da leitura do df
    '''
    n = features.shape[0] 
    featuresSplit = np.zeros((n, 100, 3))

    featuresSplit[:,:,0] = features[:,   0:100]
    featuresSplit[:,:,1] = features[:, 100:200]
    featuresSplit[:,:,2] = features[:, 200:300]

    new_features = np.zeros((n, 100, 3))
    
    new_features[:, 0:20, :]   = ( 255 * mat2gray(featuresSplit[:, 0:20, :])   ).astype(np.uint8)
    new_features[:, 20:40 ,:]  = ( 255 * mat2gray(featuresSplit[:, 20:40, :])  ).astype(np.uint8)
    new_features[:, 40:60 ,:]  = ( 255 * mat2gray(featuresSplit[:, 40:60, :])  ).astype(np.uint8)
    new_features[:, 60:80 ,:]  = ( 255 * mat2gray(featuresSplit[:, 60:80, :])  ).astype(np.uint8)
    new_features[:, 80:100 ,:] = ( 255 * mat2gray(featuresSplit[:, 80:100, :]) ).astype(np.uint8)


    if out_paths is None:
        path = f'{destino}/F-Classical'
        os.makedirs(path, exist_ok=True)

        for i in range(n):
            img = new_features[i, :, :].reshape(10, 10, 3)
            img = img.astype(np.uint8)
            img = np.fliplr(img)
            img = np.rot90(img, k=1)

            iio.imwrite(f'{path}/F-Classical{i+1}.png', img)
    else:
        for i in range(n):
            img = new_features[i, :, :].reshape(10, 10, 3)
            img = img.astype(np.uint8)
            img = np.fliplr(img)
            img = np.rot90(img, k=1)
            out_path = out_paths[i]
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            iio.imwrite(out_path, img)