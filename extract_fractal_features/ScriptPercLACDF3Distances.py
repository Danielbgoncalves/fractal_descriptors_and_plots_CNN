import glob
import time
import os
from PIL import Image
import numpy as np
import pandas as pd

from extract_fractal_features.clustperc import clustperc, clustperc_jit, clustperc_jit_plus
from extract_fractal_features.clustpercEucl import clustpercEucl, clustpercEucl_jit, clustpercEucl_jit_plus
from extract_fractal_features.clustpercManh import clustpercManh, clustpercManh_jit, clustpercManh_jit_plus
from extract_fractal_features.lacunaridade import lacunaridade, lacunaridade_plus
from extract_fractal_features.N import N
from extract_fractal_features.pmr import pmr, pmr_plus
from extract_fractal_features.pmrEucl import pmrEucl, pmrEucl_plus
from extract_fractal_features.pmrManh import pmrManh, pmrManh_plus
from scipy.stats import skew
from sklearn.linear_model import HuberRegressor

def scriptPercLACDF3Distances(diretorio_org):
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

    lista_de_resultados = []

    nome_da_classe = os.path.basename(diretorio_org)
    print(f'{len(imagens)} imagens encontradas')
    print('Coletando Descritores fractais das imagens da pasta - ', nome_da_classe )

    for n, caminho in enumerate(imagens):

        print(f"Calculando Percolação ({n+1} / {len(imagens)})")

        img_pil = Image.open(caminho)
        img_pil_resized = img_pil.resize((224, 224), Image.BILINEAR)
        PIC = np.array(img_pil_resized)

        Minsk_perc = clustperc_jit_plus(PIC, maxr)
        Eucl_perc = clustpercEucl_jit_plus(PIC, maxr)
        Manh_perc = clustpercManh_jit_plus(PIC, maxr)
        
        resultado_parc = {**Minsk_perc, **Eucl_perc, **Manh_perc}

        # --- LAC and DF (following ScriptLACDF3Distances.py) ---
        print(f"Calculando DF e LAC Minkowski {n+1} / {len(imagens)}")

        # Minkowski (Chessboard)
        MatrizProb = pmr_plus(PIC, maxr)
        MinkLAC = lacunaridade_plus(MatrizProb)
        resultado_parc['MinkLAC'] = MinkLAC
        r = list(range(3, maxr + 1, 2))
        resultado_parc['MinkAreaLAC'] = np.trapz(MinkLAC)
        resultado_parc['MinkSkewnessLAC'] = skew(MinkLAC)
        half = int( np.ceil(len(MinkLAC)/2) )
        resultado_parc['MinkAreaRatioLAC'] = np.trapz(MinkLAC[half:]) / np.trapz(MinkLAC[:half])
        resultado_parc['MinkMaxLAC'], resultado_parc['MinkMaxLACIndex'] = np.max(MinkLAC), np.argmax(MinkLAC)
        Minknn = N(MatrizProb)
        resultado_parc['Minknn'] = Minknn
        x = np.log(r)
        y = -np.log(Minknn)
        X = x.reshape(-1, 1)
        modelo = HuberRegressor()
        modelo.fit(X,y)
        resultado_parc['MinkDF'] = modelo.coef_[0]

        # Euclidean
        print(f"Calculando DF e LAC Euclidean {n+1} / {len(imagens)}")
        MatrizProb = pmrEucl_plus(PIC, maxr)
        EuclLAC = lacunaridade_plus(MatrizProb)
        resultado_parc['EuclLAC'] = EuclLAC
        r = list(range(3, maxr+1, 2))
        resultado_parc['EuclAreaLAC'] = np.trapz(EuclLAC)
        resultado_parc['EuclSkewnessLAC'] = skew(EuclLAC)
        half = int( np.ceil(len(EuclLAC)/2) )
        resultado_parc['EuclAreaRatioLAC'] = np.trapz(EuclLAC[half:]) / np.trapz(EuclLAC[:half])
        resultado_parc['EuclMaxLAC'], resultado_parc['EuclMaxLACIndex'] = np.max(EuclLAC), np.argmax(EuclLAC)
        Euclnn = N(MatrizProb)
        resultado_parc['Euclnn'] = Euclnn
        x = np.log(r)
        y = -np.log(Euclnn)
        X = x.reshape(-1, 1)
        modelo = HuberRegressor()
        modelo.fit(X,y)
        resultado_parc['EuclDF'] = modelo.coef_[0]

        # Manhattan
        print(f"Calculando DF e LAC Manhattan {n+1} / {len(imagens)}")
        MatrizProb = pmrManh_plus(PIC, maxr)
        ManhLAC = lacunaridade_plus(MatrizProb)
        resultado_parc['ManhLAC'] = ManhLAC
        r = list(range(3, maxr+1, 2))
        resultado_parc['ManhAreaLAC'] = np.trapz(ManhLAC)
        resultado_parc['ManhSkewnessLAC'] = skew(ManhLAC)
        half = int( np.ceil(len(ManhLAC)/2) )
        resultado_parc['ManhAreaRatioLAC'] = np.trapz(ManhLAC[half:]) / np.trapz(ManhLAC[:half])
        resultado_parc['ManhMaxLAC'], resultado_parc['ManhMaxLACIndex'] = np.max(ManhLAC), np.argmax(ManhLAC)
        Manhnn = N(MatrizProb)
        resultado_parc['Manhnn'] = Manhnn
        x = np.log(r)
        y = -np.log(Manhnn)
        X = x.reshape(-1, 1)
        modelo = HuberRegressor()
        modelo.fit(X,y)
        resultado_parc['ManhDF'] = modelo.coef_[0]

        lista_de_resultados.append(resultado_parc)

        
    # retornar como DataFrame com todas as colunas (cada campo é uma coluna);
    try:
        df = pd.DataFrame(lista_de_resultados)
        return df
    except Exception:
        return lista_de_resultados

