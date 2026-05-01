import os
import glob
import time
import pandas as pd
import numpy as np
import subprocess
import sys
import shutil

from extract_fractal_features.util import reorganizar_e_expandir_df
from extract_fractal_features.ScriptPercLACDF3Distances import scriptPercLACDF3Distances

"""
Main Feature Extraction Pipeline

Este arquivo é o ponto central de execução de todo o repositório.

Ele orquestra o pipeline completo de processamento de imagens, realizando:
1. Varredura recursiva de diretórios contendo imagens
2. Extração de descritores através do algoritmo PercLACDF3Distances
3. Organização e padronização dos dados em DataFrames
4. Expansão e transformação dos descritores para formato tabular
5. Geração de arquivos CSV (com e sem metadados)
6. Conversão dos descritores em representações visuais (imagens)
7. Reconstrução da estrutura original do dataset para os outputs gerados

Saída:
    - CSV com descritores 
    - CSV com descritores + metadados 
    - Imagens geradas a partir dos descritores
    - Estrutura de saída espelhando a organização original dos dados

Exemplo de uso:
    python extract_features_and_generate_outputs <origem> <destino> [mode]

Parâmetros:
    origem  : Caminho para o dataset de entrada
    destino : Caminho onde os resultados serão salvos
    mode    : Tipo de imagem gerada ('both', 'recplot', 'classical')

Observação:
    A consistência entre a ordem das imagens e os descritores retornados
    é essencial para garantir o correto mapeamento dos resultados.
"""


def collect_image_files_in_dir(directory):
    padrao_png = os.path.join(directory, '*.png').replace('\\', '/')
    padrao_tif = os.path.join(directory, '*.tif').replace('\\', '/')
    padrao_jpg = os.path.join(directory, '*.jpg').replace('\\', '/')
    imgs = glob.glob(padrao_png) + glob.glob(padrao_tif) + glob.glob(padrao_jpg)
    return imgs


def extract_features_and_generate_outputs(origem, destino, mode='both'):
    tic = time.time()

    origem = os.path.abspath(origem)
    destino = os.path.abspath(destino)
    os.makedirs(destino, exist_ok=True)

    resultados_completos = []

    # Procurar por subpastas que contenham imagens 
    for root, dirs, files in os.walk(origem):
        imagens = collect_image_files_in_dir(root)
        if not imagens:
            continue

        print(f"Processando pasta: {root} ({len(imagens)} imagens)")
        resultado = scriptPercLACDF3Distances(root)

        # garantir lista de dicts
        if isinstance(resultado, pd.DataFrame):
            df_res = resultado
            lista_res = df_res.to_dict(orient='records')
        else:
            lista_res = list(resultado)

        imagens_ord = imagens

        # associar cada entrada com sua imagem e subpasta relativa
        rel_subfolder = os.path.relpath(root, origem)
        # determinar nome da pasta de origem (mesmo quando rel_subfolder == '.')
        if rel_subfolder == '.':
            top_folder = os.path.basename(root)
            rel_subfolder_norm = top_folder
        else:
            rel_subfolder_norm = rel_subfolder.replace('\\', '/')

        for i, item in enumerate(lista_res):
            entry = dict(item)
            # se houver correspondência de nomes, anexe; senão use índice
            try:
                img_path = imagens_ord[i]
            except IndexError:
                img_path = None

            # Não deveria ser None em momento algum. Se sair None é um problema, confira os prints anteriores para entender o que aconteceu.
            entry['image_name'] = os.path.basename(img_path) if img_path else None
            entry['subfolder'] = rel_subfolder_norm

            if entry['image_name']:
                rel_path = os.path.join(rel_subfolder_norm, entry['image_name']).replace('\\', '/')
            else:
                rel_path = None
            entry['relative_path'] = rel_path

            resultados_completos.append(entry)

    df_full = pd.DataFrame(resultados_completos)

    # converter arrays numpy para listas para salvar corretamente
    for col in df_full.columns:
        df_full[col] = df_full[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    # gerar DataFrame expandido/organizado a partir do DataFrame completo
    try:
        df_full_expanded = reorganizar_e_expandir_df(df_full)
    except Exception as e:
        print('Erro ao reorganizar/expandir o dataframe, esse processo garante que esteja no formato esperado:', e)
        df_full_expanded = df_full.copy()

    # garantir que `image_name` e `subfolder` existam e estejam alinhados
    if 'image_name' not in df_full_expanded.columns:
        df_full_expanded['image_name'] = df_full.get('image_name')
    if 'subfolder' not in df_full_expanded.columns:
        df_full_expanded['subfolder'] = df_full.get('subfolder')

    # 1) CSV com APENAS os descritores (expandidos/reorganizados)
    try:
        df_results = reorganizar_e_expandir_df(df_full)
    except Exception as e:
        print('Erro ao gerar df_results (reorganizar):', e)
        df_results = df_full_expanded.copy()

    # remover colunas de identificação, se existirem
    df_results_only = df_results.copy()
    if 'image_name' in df_results_only.columns:
        df_results_only = df_results_only.drop(columns=['image_name'])
    if 'subfolder' in df_results_only.columns:
        df_results_only = df_results_only.drop(columns=['subfolder'])

    caminho_csv_results = os.path.join(destino, 'result_perc_lacdf.csv')
    df_results_only.to_csv(caminho_csv_results, index=False, sep=';', decimal='.')

    # 2) CSV com TODOS os dados expandidos e as colunas `image_name` e `subfolder`
    df_with_paths = df_full_expanded.copy()
    if 'image_name' not in df_with_paths.columns:
        df_with_paths['image_name'] = df_full.get('image_name')
    if 'subfolder' not in df_with_paths.columns:
        df_with_paths['subfolder'] = df_full.get('subfolder')

    caminho_csv_with_paths = os.path.join(destino, 'result_perc_lacdf_with_paths.csv')
    df_with_paths.to_csv(caminho_csv_with_paths, index=False, sep=';', decimal='.')

    # Gerar CSV com features + relative_path + image_name para create_imgs
    csv_for_imgs = os.path.join(destino, 'tmp_features_for_imgs.csv')
    # df_results_only contém apenas os descritores expandidos; precisamos anexar image_name e relative_path
    df_for_imgs = df_results_only.copy()
    # garantir mesma ordem: usar df_full (tem relative_path e image_name originais)
    if 'relative_path' in df_full.columns and 'image_name' in df_full.columns:
        df_for_imgs['relative_path'] = df_full['relative_path'].values
        df_for_imgs['image_name'] = df_full['image_name'].values
    else:
        # fallback para evitar exceção — criar colunas vazias
        df_for_imgs['relative_path'] = [None] * len(df_for_imgs)
        df_for_imgs['image_name'] = [None] * len(df_for_imgs)
    df_for_imgs.to_csv(csv_for_imgs, index=False, sep=',')

    create_imgs_py = os.path.join(os.path.dirname(__file__), 'feature_to_image', 'create_imgs.py')
    mirrored_base = os.path.join(destino, 'mirrored_images')
    os.makedirs(mirrored_base, exist_ok=True)

    try:
        subprocess.run([sys.executable, create_imgs_py, csv_for_imgs, mirrored_base, mode], check=True)
    except Exception as e:
        print('Erro ao executar create_imgs.py:', e)


    # remover CSV temporário
    try:
        if os.path.exists(csv_for_imgs):
            os.remove(csv_for_imgs)
    except Exception:
        pass

    toc = time.time()
    print(f"Salvo em {destino}")
    print(f"Tempo total: {toc - tic:.2f}s")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Uso: python main_pipeline.py <origem_path> <destino_path> [mode: both|recplot|classical]')
        sys.exit(1)
    origem = sys.argv[1]
    destino = sys.argv[2]
    mode = sys.argv[3].lower() if len(sys.argv) > 3 else 'both'
    extract_features_and_generate_outputs(origem, destino, mode=mode)
