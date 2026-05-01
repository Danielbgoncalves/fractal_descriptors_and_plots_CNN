import glob
import time
import os
from PIL import Image
import numpy as np
import pandas as pd

from clustperc import clustperc, clustperc_jit
from clustpercEucl import clustpercEucl, clustpercEucl_jit
from clustpercManh import clustpercManh, clustpercManh_jit

def scriptPerc(diretorio_org):
    '''
    Extração de atributos DF LAC 
    com uso das métricas de distância Chessboard, Euclidiana e Manhatan a partir de imagens RGB.
    
    Daniel Borges gonçalves
    Outubro de 2025
    '''

    maxr = 41

    padrao_png = os.path.join(diretorio_org, '*.png').replace("\\", "/")
    padrao_tif = os.path.join(diretorio_org, '*.tif').replace("\\", "/")
    padrao_jpg = os.path.join(diretorio_org, '*.jpg').replace("\\", "/")

    imagens = glob.glob(padrao_png) + glob.glob(padrao_tif) + glob.glob(padrao_jpg)

    df = pd.DataFrame({
        "nome_original": imagens,
        "nome_recplot": [f"{i}.png" for i in range(1, len(imagens) + 1)]
    })

    df.to_csv("mapeamento_pulmao/aca.csv", index=False)

    lista_de_resultados = []

    nome_da_classe = os.path.basename(diretorio_org)
    print(f'{len(imagens)} encontradas')
    print('Coletando características Fractais de Percolaçãodas da pasta - ', nome_da_classe )

    for n, caminho in enumerate(imagens):

        print(f"Calculando Percolação ({n+1} / {len(imagens)})")

        img_pil = Image.open(caminho)
        img_pil_resized = img_pil.resize((224, 224), Image.BILINEAR)
        PIC = np.array(img_pil_resized)

        Minsk_perc = clustperc_jit(PIC, maxr)
        Eucl_perc = clustpercEucl_jit(PIC, maxr)
        Manh_perc = clustpercManh_jit(PIC, maxr)
        
        resultado_parc = {**Minsk_perc, **Eucl_perc, **Manh_perc}
        lista_de_resultados.append(resultado_parc)
        
    return lista_de_resultados

