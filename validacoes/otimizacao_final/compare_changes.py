import glob
import os
import time
from PIL import Image
import numpy as np
import pandas as pd
import gc

from extract_fractal_features.clustperc import clustperc_jit_plus
from extract_fractal_features.clustpercEucl import clustpercEucl_jit_plus
from extract_fractal_features.clustpercManh import clustpercManh_jit_plus
from extract_fractal_features.lacunaridade import lacunaridade_plus
from extract_fractal_features.N import N
from extract_fractal_features.pmr import pmr_plus
from extract_fractal_features.pmrEucl import pmrEucl_plus
from extract_fractal_features.pmrManh import pmrManh_plus
from sklearn.linear_model import HuberRegressor

from extract_fractal_features.util import reorganizar_e_expandir_df_benchmark 

'''
    ############## Aqui Mudanças serão testadas ##############

    Os scripts clusterperc continuam não retornando os valores originais.
    Apenas p, g e h serão retornados. 

    Lembrar de descomentar o retorno destes scripts quando for usar fora do benchmark!
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

def gerar_comparacao(diretorio_org):
    maxr = 41
    resultados_numericos = []
    resultados_tempo = []
    os.makedirs('benchmark_compare', exist_ok=True)
    os.makedirs('benchmark_compare/MatrizProb', exist_ok=True)
    os.makedirs('benchmark_compare/MatrizProbEucl', exist_ok=True)
    os.makedirs('benchmark_compare/MatrizProbManh', exist_ok=True)

    padrao_png = os.path.join(diretorio_org, '*.png').replace("\\", "/")
    padrao_tif = os.path.join(diretorio_org, '*.tif').replace("\\", "/")
    padrao_jpg = os.path.join(diretorio_org, '*.jpg').replace("\\", "/")

    imagens = glob.glob(padrao_png) + glob.glob(padrao_tif) + glob.glob(padrao_jpg)

    print(f'{len(imagens)} imagens encontradas')

    print('# Iniciando Warm-up')
    PIC_dummy = np.ones((224,224,3), dtype=np.uint8)
    ProbMatriz_dummy = np.ones((1681, 20), dtype=np.float64)
    _ = clustperc_jit_plus(PIC_dummy, maxr)
    _ = clustpercEucl_jit_plus(PIC_dummy, maxr)
    _ = clustpercManh_jit_plus(PIC_dummy, maxr)
    _ = pmr_plus(PIC_dummy, maxr)
    _ = pmrEucl_plus(PIC_dummy, maxr)
    _ = pmrManh_plus(PIC_dummy, maxr)
    _ = lacunaridade_plus(ProbMatriz_dummy)
    nn = N(ProbMatriz_dummy)
    _ = calcula_df(maxr, nn)
    print('# Warm-up concluído. Iniciando contagem real.')
    
    for n, caminho in enumerate(imagens):

        img_pil = Image.open(caminho)
        img_pil_resized = img_pil.resize((224, 224), Image.BILINEAR)
        PIC = np.array(img_pil_resized)
        img_id = os.path.basename(caminho)

        resultado_parc_n = {} # resultados numéricos
        resultado_parc_t = {} # tempo gasto
        resultado_parc_n['img_id'] = img_id
        resultado_parc_t['img_id'] = img_id

        #--- Percolação ---
        resultado, tempo = executar_e_medir(clustperc_jit_plus, PIC, maxr)
        resultado_parc_n.update(resultado)
        resultado_parc_t['Mink_perc'] = tempo

        resultado, tempo = executar_e_medir(clustpercEucl_jit_plus, PIC, maxr)
        resultado_parc_n.update(resultado)
        resultado_parc_t['Eucl_perc'] = tempo

        resultado, tempo = executar_e_medir(clustpercManh_jit_plus, PIC, maxr)
        resultado_parc_n.update(resultado)
        resultado_parc_t['Manh_perc'] = tempo

        #--- LAC and DF ---
        # Minkowski
        MatrizProb, tempo = executar_e_medir(pmr_plus, PIC, maxr)
        np.save(f'benchmark_compare/MatrizProb/matriz_prob_{img_id}.npy', MatrizProb)
        resultado_parc_t[f'MatrizProb'] = tempo

        # Euclidean
        MatrizProbEucl, tempo = executar_e_medir(pmrEucl_plus, PIC, maxr)
        np.save(f'benchmark_compare/MatrizProbEucl/matriz_prob_eucl_{img_id}.npy', MatrizProbEucl)
        resultado_parc_t[f'MatrizProbEucl'] = tempo

        # Manhattan
        MatrizProbManh, tempo = executar_e_medir(pmrManh_plus, PIC, maxr)
        np.save(f'benchmark_compare/MatrizProbManh/matriz_prob_manh_{img_id}.npy', MatrizProbManh)
        resultado_parc_t[f'MatrizProbManh'] = tempo

        # Como lacunaridade e N são calculados a partir da matriz de probabilidade, 
        # eles serão calculados para uma única métrica de distância
        # Lacunaridade
        lac_result, tempo = executar_e_medir(lacunaridade_plus, MatrizProb)
        resultado_parc_n['LAC'] = lac_result
        resultado_parc_t['LAC'] = tempo
        
        # Função N (Contagem de caixas)
        nn, tempo = executar_e_medir(N, MatrizProb)
        resultado_parc_n['nn'] = nn
        resultado_parc_t['nn'] = tempo
        
        # #--- Dimensão Fractal ---
        df, tempo = executar_e_medir(calcula_df, maxr, nn)
        resultado_parc_n['DF'] = df
        resultado_parc_t[f'DF'] = tempo


        resultados_numericos.append(resultado_parc_n)
        resultados_tempo.append(resultado_parc_t)
    
    df_n = pd.DataFrame(resultados_numericos)
    df_n = reorganizar_e_expandir_df_benchmark(df_n)

    df_t = pd.DataFrame(resultados_tempo)

    return df_n, df_t

def compare_baseline_opt(base_path, opt_path):
    df_base = pd.read_csv(base_path)
    df_opt = pd.read_csv(opt_path)

    common_cols = list(set(df_base.columns) & set(df_opt.columns))
    if 'img_id' in common_cols:
        common_cols.remove('img_id')

    print(f"Número de imagens no baseline: {len(df_base)}")
    print(f"Número de imagens no otimizado: {len(df_opt)}")
    print(f"Número de colunas em comum analisadas: {len(common_cols)}")

    if len(common_cols) == 0: return

    df_merged = pd.merge(df_base[['img_id'] + common_cols], df_opt[['img_id'] + common_cols], on='img_id', suffixes=('_base', '_opt'))

    results = []
    for col in common_cols:
        base_vals = df_merged[f'{col}_base']
        opt_vals = df_merged[f'{col}_opt']
        
        # Calcular a diferença absoluta
        diff = np.abs(base_vals - opt_vals)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        results.append({
            'Coluna': col,
            'Diferenca_Maxima': max_diff,
            'Diferenca_Media': mean_diff
        })

    df_results = pd.DataFrame(results)
    # Ordenar pelas maiores diferenças
    df_results = df_results.sort_values(by='Diferenca_Maxima', ascending=False)

    print("\n--- Top 15 Maiores Diferenças Máximas ---")
    print(df_results.head(15).to_string(index=False))

    print("\n--- Resumo Estatístico das Diferenças Máximas ---")
    print(df_results['Diferenca_Maxima'].describe())

    # perfect matches..
    perfect_matches = (df_results['Diferenca_Maxima'] < 1e-10).sum()
    print(f"\nColunas com match perfeito (diferença < 1e-10): {perfect_matches} de {len(common_cols)}")

def compare_prog_matrix(base_path, opt_path):
    
    padrao = os.path.join(base_path, '*.npy').replace("\\", "/")
    arquivos_base = glob.glob(padrao)
    
    if not arquivos_base:
        print("Nenhum arquivo .npy encontrado na pasta baseline!")
        return

    resultados = []

    for caminho_base in arquivos_base:
        nome_arquivo = os.path.basename(caminho_base)
        
        caminho_opt = os.path.join(opt_path, nome_arquivo).replace("\\", "/")
        
        if not os.path.exists(caminho_opt):
            print(f"Aviso: Arquivo {nome_arquivo} não encontrado na pasta otimizada. Pulando...")
            continue
            
        base_mat = np.load(caminho_base)
        opt_mat = np.load(caminho_opt)
        
        if base_mat.shape != opt_mat.shape:
            print(f"Erro em {nome_arquivo}: Shapes diferentes! Base {base_mat.shape} vs Opt {opt_mat.shape}")
            continue
            
        diff = np.abs(base_mat - opt_mat)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # np.allclose verifica se todos os elementos são iguais dentro de uma tolerância (1e-10)
        sao_iguais = np.allclose(base_mat, opt_mat, atol=1e-10)
        
        resultados.append({
            'Arquivo': nome_arquivo,
            'Diff_Maxima': max_diff,
            'Diff_Media': mean_diff,
            'Aprovado': sao_iguais
        })
        
    # 6. Exibe um relatório bonitão usando o Pandas
    if resultados:
        df_relatorio = pd.DataFrame(resultados)
        print("=== RELATÓRIO DE COMPARAÇÃO DE MATRIZES ===")
        print(df_relatorio.to_string(index=False))
        
        total = len(df_relatorio)
        aprovados = df_relatorio['Aprovado'].sum()
        
        print("\n=== RESUMO FINAL ===")
        print(f"Matrizes analisadas: {total}")
        print(f"Matrizes idênticas (tolerância 1e-10): {aprovados}/{total}")
        
        if aprovados == total:
            print("SUCESSO! A otimização preservou a matemática perfeitamente.")
        else:
            print("ATENÇÃO! Algumas matrizes divergiram. Verifique a coluna 'Diff_Maxima'.")

def comparar_tempos_csv(caminho_baseline, caminho_otimizado):
    print("Carregando arquivos de tempo...")
    df_base = pd.read_csv(caminho_baseline)
    df_opt = pd.read_csv(caminho_otimizado)

    # 1. Encontra apenas as colunas de tempo que existem nos dois arquivos
    colunas_comuns = list(set(df_base.columns) & set(df_opt.columns))
    if 'img_id' in colunas_comuns:
        colunas_comuns.remove('img_id')

    if not colunas_comuns:
        print("Erro: Nenhuma coluna de tempo em comum encontrada além do img_id.")
        return

    # 2. Junta os dois DataFrames lado a lado usando a imagem como chave
    df_merged = pd.merge(
        df_base[['img_id'] + colunas_comuns], 
        df_opt[['img_id'] + colunas_comuns], 
        on='img_id', 
        suffixes=('_base', '_opt')
    )

    resultados = []

    # 3. Calcula as métricas para cada função/etapa
    for col in colunas_comuns:
        tempos_base = df_merged[f'{col}_base']
        tempos_opt = df_merged[f'{col}_opt']

        # Calculamos a média de tempo gasto na etapa (ignorando variações de uma imagem específica)
        media_base = tempos_base.mean()
        media_opt = tempos_opt.mean()
        
        # O quanto de tempo bruto foi economizado por imagem
        tempo_economizado = media_base - media_opt
        
        # Speedup = Quantas vezes mais rápido ficou (Ex: 2.0x, 10.5x)
        # Usamos np.where ou if para evitar dividir por zero caso a função otimizada tenha zerado o relógio
        if media_opt > 0:
            speedup = media_base / media_opt
        else:
            speedup = np.nan
            
        # Redução percentual (Ex: reduziu 95% do tempo)
        if media_base > 0:
            reducao_pct = (tempo_economizado / media_base) * 100
        else:
            reducao_pct = np.nan

        resultados.append({
            'Função/Etapa': col,
            'Base_Média(s)': round(media_base, 4),
            'Opt_Média(s)': round(media_opt, 4),
            'Economia(s)': round(tempo_economizado, 4),
            'Speedup (Vezes)': round(speedup, 2),
            'Redução (%)': round(reducao_pct, 2)
        })

    # 4. Cria o DataFrame final e ordena para os maiores ganhos aparecerem no topo
    df_relatorio = pd.DataFrame(resultados)
    df_relatorio = df_relatorio.sort_values(by='Speedup (Vezes)', ascending=False)

    print("\n" + "="*70)
    print("🏆 RELATÓRIO DE OTIMIZAÇÃO DE PERFORMANCE (SPEEDUP) 🏆")
    print("="*70)
    print(df_relatorio.to_string(index=False))
    print("="*70)
    
    # Salva o relatório para você usar no TCC
    df_relatorio.to_csv('relatorio_ganho_tempo.csv', index=False)
    print("\nRelatório salvo como 'relatorio_ganho_tempo.csv'")
    
    return df_relatorio

#--- Coleta imagens e guarda resultados e tempos de cada etapa
# diretorio_org = r'entradas_para_testes/median'
# df_n, df_t = gerar_comparacao(diretorio_org)

# df_n.to_csv('benchmark_compare/completo_valores.csv', index=False)
# df_t.to_csv('benchmark_compare/completo_tempo.csv', index=False)

#--- Compara os resultados salvos em csv e printa no terminal
compare_baseline_opt(r'benchmark\benchmark_baseline\benchmark_valores.csv',r'benchmark\benchmark_opt\completo_valores.csv')

#--- Compara os resultados das matrizes salvas como .npy e printa no terminal
compare_prog_matrix(r'benchmark/benchmark_baseline/MatrizProb', r'benchmark/benchmark_opt/MatrizProb')
compare_prog_matrix(r'benchmark/benchmark_baseline/MatrizProbEucl', r'benchmark/benchmark_opt/MatrizProbEucl')
compare_prog_matrix(r'benchmark/benchmark_baseline/MatrizProbManh', r'benchmark/benchmark_opt/MatrizProbManh')

#--- Compara o Speedup com csv de tempos de cada etapa
# comparar_tempos_csv('benchmark_baseline/benchmark_tempo.csv', 'benchmark_compare/completo_tempo.csv')

