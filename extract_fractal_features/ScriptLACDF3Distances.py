import glob
import os
import time
from PIL import Image
import numpy as np
from scipy.stats import skew
from sklearn.linear_model import HuberRegressor
import pandas as pd

from lacunaridade import lacunaridade 
from N import N

from pmr import pmr
from pmrEucl import pmrEucl
from pmrManh import pmrManh

def scriptLACDF3Distances(diretorio_org):
    '''
    Extração de atributos DF LAC
    com uso das métricas de distância Chessboard, Euclidiana e Manhatan a partir de imagens RGB.
    Seu resultado é um dicionário ao invés do arquivo .mat usado na versão MATLAB

    ######################

    Daniel Borges gonçalves
    Outubro de 2025

    ######################
    Essa é uma tradução direta, veratoração com loop evitaria repetição desnecessária e deixaria mais limpo
    '''

    # valor máximo de L
    maxr = 41

    padrao_png = os.path.join(diretorio_org, '*.png').replace("\\", "/")
    padrao_tif = os.path.join(diretorio_org, '*.tif').replace("\\", "/")
    padrao_jpg = os.path.join(diretorio_org, '*.jpg').replace("\\", "/")

    imagens = glob.glob(padrao_png) + glob.glob(padrao_tif) + glob.glob(padrao_jpg)

    lista_de_resultados = []

    nome_da_classe = os.path.basename(diretorio_org)
    print('Coletando LAC e DF das imagens da pasta - ', nome_da_classe )


    for n, caminho in enumerate(imagens):
        n += 1
        resultado_parc = {}

        fullname = caminho
        img_pil = Image.open(caminho)
        img_pil = img_pil.resize((224, 224), Image.BILINEAR)
        PIC = np.array(img_pil)


        print(f"Calculando características fractais - Minkowski({n} / {len(imagens)})")
        MatrizProb = pmr(PIC, maxr)
        MinkLAC = lacunaridade(MatrizProb)
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
        X = x.reshape(-1, 1) # ou x.T eu acho
        #Y = y.reshape(-1, 1)
        modelo = HuberRegressor()
        modelo.fit(X,y)
        resultado_parc['MinkDF'] = modelo.coef_[0]

        print(f"Calculando características fractais - Euclidian({n} / {len(imagens)})")
        MatrizProb = pmrEucl(PIC, maxr)
        EuclLAC = lacunaridade(MatrizProb)
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
        X = x.reshape(-1, 1) # ou x.T eu acho
        #Y = y.reshape(-1, 1)
        modelo = HuberRegressor()
        modelo.fit(X,y)
        resultado_parc['EuclDF'] = modelo.coef_[0]

        print(f"Calculando características fractais - Manhattan({n} / {len(imagens)})")
        MatrizProb = pmrManh(PIC, maxr)
        ManhLAC = lacunaridade(MatrizProb)
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
        X = x.reshape(-1, 1) # ou x.T eu acho
        #Y = y.reshape(-1, 1)
        modelo = HuberRegressor()
        modelo.fit(X,y)
        resultado_parc['ManhDF'] = modelo.coef_[0]

        lista_de_resultados.append(resultado_parc)    

    return lista_de_resultados

    
    
