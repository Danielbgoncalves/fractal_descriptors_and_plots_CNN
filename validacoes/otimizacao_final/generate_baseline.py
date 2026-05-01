import glob
import os
import time
from PIL import Image
import numpy as np
import pandas as pd
import gc

from extract_fractal_features.clustperc import clustperc, clustperc_jit
from extract_fractal_features.clustpercEucl import clustpercEucl, clustpercEucl_jit
from extract_fractal_features.clustpercManh import clustpercManh, clustpercManh_jit
from extract_fractal_features.lacunaridade import lacunaridade
from extract_fractal_features.N import N
from extract_fractal_features.pmr import pmr
from extract_fractal_features.pmrEucl import pmrEucl
from extract_fractal_features.pmrManh import pmrManh
from sklearn.linear_model import HuberRegressor

from extract_fractal_features.util import reorganizar_e_expandir_df_benchmark 

'''
    ############## Benchmark ##############

    Os scripts clusterperc não irão retornar os valores originais
    para o benchmark, apenas p, g e h serão retornados 
    (os demias são deduzidos destes)

    Lembrar de descomentar o retorno destes scripts quando for usar fora do benchmark
'''

def executar_e_medir(funcao, *args):

    gc.collect()
    gc.disable()

    tic = time.perf_counter()
    resultado = funcao(*args)
    toc = time.perf_counter()

    tempo = toc - tic

    gc.enable()
    
    return resultado, tempo
    
def calcula_df(maxr, nn):
    r = list(range(3, maxr+1, 2))
    x = np.log(r)
    y = -np.log(nn)
    X = x.reshape(-1, 1)
    modelo = HuberRegressor()
    modelo.fit(X,y)
    return modelo.coef_[0]

def gerar_baseline(diretorio_org):
    maxr = 41
    resultados_numericos = []
    resultados_tempo = []
    os.makedirs('benchmark_baseline', exist_ok=True)
    os.makedirs('benchmark_baseline/MatrizProb', exist_ok=True)
    os.makedirs('benchmark_baseline/MatrizProbEucl', exist_ok=True)
    os.makedirs('benchmark_baseline/MatrizProbManh', exist_ok=True)


    padrao_png = os.path.join(diretorio_org, '*.png').replace("\\", "/")
    padrao_tif = os.path.join(diretorio_org, '*.tif').replace("\\", "/")
    padrao_jpg = os.path.join(diretorio_org, '*.jpg').replace("\\", "/")

    imagens = glob.glob(padrao_png) + glob.glob(padrao_tif) + glob.glob(padrao_jpg)


    print(f'{len(imagens)} imagens encontradas')

    print('Iniciando Warm-up')
    PIC_dummy = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    ProbMatriz_dummy = np.ones((1681, 20), dtype=np.float64)
    _ = clustperc_jit(PIC_dummy, maxr)
    _ = clustpercEucl_jit(PIC_dummy, maxr)
    _ = clustpercManh_jit(PIC_dummy, maxr)
    _ = pmr(PIC_dummy, maxr)
    _ = pmrEucl(PIC_dummy, maxr)
    _ = pmrManh(PIC_dummy, maxr)
    _ = lacunaridade(ProbMatriz_dummy)
    nn = N(ProbMatriz_dummy)
    _ = calcula_df(maxr, nn)
    print('Warm-up concluído.\nIniciando contagem real.')
    
    for n, caminho in enumerate(imagens):

        img_pil = Image.open(caminho)
        img_pil_resized = img_pil.resize((224, 224), Image.BILINEAR)
        PIC = np.array(img_pil_resized)
        img_id = os.path.basename(caminho)

        resultado_parc_n = {} # resultados numéricos
        resultado_parc_t = {} # tempo gasto
        resultado_parc_n['img_id'] = img_id
        resultado_parc_t['img_id'] = img_id

        #--- Percolação (following ScriptPercLACDF3Distances.py) ---
        resultado, tempo = executar_e_medir(clustperc_jit, PIC, maxr)
        resultado_parc_n.update(resultado)
        resultado_parc_t['Mink_perc'] = tempo

        resultado, tempo = executar_e_medir(clustpercEucl_jit, PIC, maxr)
        resultado_parc_n.update(resultado)
        resultado_parc_t['Eucl_perc'] = tempo

        resultado, tempo = executar_e_medir(clustpercManh_jit, PIC, maxr)
        resultado_parc_n.update(resultado)
        resultado_parc_t['Manh_perc'] = tempo

        #--- LAC and DF (following ScriptLACDF3Distances.py) ---
        # Minkowski
        MatrizProb, tempo = executar_e_medir(pmr, PIC, maxr)
        np.save(f'benchmark_baseline/MatrizProb/matriz_prob_{img_id}.npy', MatrizProb)
        resultado_parc_t[f'MatrizProb'] = tempo

        # Euclidean
        MatrizProbEucl, tempo = executar_e_medir(pmrEucl, PIC, maxr)
        np.save(f'benchmark_baseline/MatrizProbEucl/matriz_prob_eucl_{img_id}.npy', MatrizProbEucl)
        resultado_parc_t[f'MatrizProbEucl'] = tempo

        # Manhattan
        MatrizProbManh, tempo = executar_e_medir(pmrManh, PIC, maxr)
        np.save(f'benchmark_baseline/MatrizProbManh/matriz_prob_manh_{img_id}.npy', MatrizProbManh)
        resultado_parc_t[f'MatrizProbManh'] = tempo

        # Como lacunaridade e N são calculados a partir da matriz de probabilidade, 
        # eles serão calculados para uma única métrica de distância
        # Lacunaridade
        lac_result, tempo = executar_e_medir(lacunaridade, MatrizProb)
        resultado_parc_n['LAC'] = lac_result
        resultado_parc_t['LAC'] = tempo
        
        # Função N (Contagem de caixas)
        nn, tempo = executar_e_medir(N, MatrizProb)
        resultado_parc_n['nn'] = nn
        resultado_parc_t['nn'] = tempo
        
        #--- Dimensão Fractal (following ScriptLACDF3Distances.py) ---
        df, tempo = executar_e_medir(calcula_df, maxr, nn)
        resultado_parc_n['DF'] = df
        resultado_parc_t[f'DF'] = tempo


        resultados_numericos.append(resultado_parc_n)
        resultados_tempo.append(resultado_parc_t)
    
    df_n = pd.DataFrame(resultados_numericos)
    df_n = reorganizar_e_expandir_df_benchmark(df_n)

    df_t = pd.DataFrame(resultados_tempo)

    return df_n, df_t

diretorio_org = r'entradas_para_testes/median'
df_n, df_t = gerar_baseline(diretorio_org)

df_n.to_csv('benchmark_baseline/benchmark_valores.csv', index=False)
df_t.to_csv('benchmark_baseline/benchmark_tempo.csv', index=False)
