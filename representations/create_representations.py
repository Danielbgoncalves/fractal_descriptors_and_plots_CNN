from skimage.transform import resize
import matplotlib.pyplot as plt
import imageio.v3 as iio
import pandas as pd
import numpy as np 
import os



from .util import organize_by_nature
from .RP import generate_RP_treino, generate_RP_teste
#from .GAF import generate_GAF
#from .MTF import generate_MTF

REPRESENTATIONS_MAP = {
     'RP': (generate_RP_treino, generate_RP_teste)
    #  'GAF': generate_GAF,
    #  'MTF': generate_MTF
}

def salve_images(images, names, subfolders, base_out_path, technique, split_name):

    unique_subfolders = np.unique(subfolders)

    for subfolder in unique_subfolders:
        dir_path = os.path.join(base_out_path, technique, split_name, subfolder)
        os.makedirs(dir_path, exist_ok=True)
    
    for i in range(images.shape[0]):

        img_224 = resize(images[i], (224, 224), anti_aliasing=True)
        img_224 = np.clip(img_224, 0.0, 1.0)
        img_uint8 = ( img_224 * 255 ).astype(np.uint8)

        nome_original = names[i]
        nome_sem_extensao = os.path.splitext(nome_original)[0]
        novo_nome = f'{nome_sem_extensao}.png'

        path = os.path.join(base_out_path, technique, split_name, subfolders[i], novo_nome)
        iio.imwrite(path , img_uint8)

def create_representations(features_path, out_path, test_indices=None, plots=['RP']):
    os.makedirs(out_path, exist_ok=True)
    
    features_df = pd.read_csv(features_path, sep=';')
    print(f'O csv de entrada tem shape: {features_df.shape}\n')

    if features_df.shape[0] < 2: 
        print("Erro: O csv deve ter no mínimo duas linhas") 
        return
    
    imgs_names = features_df["image_name"].values
    imgs_subfolders = features_df["subfolder"].values

    feature_num_df = features_df.drop(columns=["image_name", "subfolder"])
    features_matrix = feature_num_df.values.astype(np.float64)

    perc, lac, massa_df = organize_by_nature(features_matrix)

    data_map = {
         'RP': perc,
        #  'GAF': lac, 
        #  'MTF': massa_df
    }

    n_amostras = features_matrix.shape[0]
    if test_indices is not None:
        mask_teste = np.zeros(n_amostras, dtype=bool)
        # O usuário passa índices amigáveis (1-based), convertemos para 0-based do Python
        for idx in test_indices:
            if 0 <= idx - 1 < n_amostras:
                mask_teste[idx - 1] = True
        mask_treino = ~mask_teste
    else:
        mask_treino = np.ones(n_amostras, dtype=bool)
        mask_teste = np.zeros(n_amostras, dtype=bool)

    for plot_type in plots:
        plot_type = plot_type.upper()

        if plot_type not in REPRESENTATIONS_MAP:
            print(f'Erro: técnica {plot_type} não implementada. Ignorando...')
            continue

        train_func, test_func = REPRESENTATIONS_MAP[plot_type]
        full_data = data_map[plot_type]

        # Processando Treino / Validacao
        X_train = full_data[mask_treino]
        if X_train.shape[0] > 0:
            print(f'[{plot_type}] Processando Treino/Validação ({X_train.shape[0]} amostras)...')
            # A função de treino gera as imagens e nos devolve os parâmetros (mins, maxs, etc.)
            train_imgs, *params = train_func(X_train)
            
            salve_images(
                images=train_imgs,
                names=imgs_names[mask_treino],
                subfolders=imgs_subfolders[mask_treino],
                base_out_path=out_path,
                technique=plot_type,
                split_name='treino_validacao'
            )
        else:
            print(f'[{plot_type}] Atenção: Nenhum dado de treino selecionado.')
            params = None

        # Processando Teste
        X_test = full_data[mask_teste]
        if X_test.shape[0] > 0:
            if params is not None:
                print(f'[{plot_type}] Processando Teste ({X_test.shape[0]} amostras) usando parâmetros do treino...')
                # Desempacota os parâmetros salvos no treino e passa para a função de teste
                test_imgs = test_func(X_test, *params)
                
                salve_images(
                    images=test_imgs,
                    names=imgs_names[mask_teste],
                    subfolders=imgs_subfolders[mask_teste],
                    base_out_path=out_path,
                    technique=plot_type,
                    split_name='teste'
                )
            else:
                print(f'[{plot_type}] Erro: Não é possível processar o teste sem parâmetros de treino.')


    print(f'Processamento concluído com sucesso em {out_path}')

if __name__ == "__main__":
    import sys

    # python create_representations.py ./saida 1-44
    # Formato esperado: python create_representations.py <csv_path> <out_path> <indices>
    if len(sys.argv) < 3:
        print("Uso: python script.py <csv_descritores> <diretorio_saida> [indices_teste]")
        sys.exit(1)

    csv_descritores = sys.argv[1]
    dir_out = sys.argv[2]

    indices_teste = None
    if len(sys.argv) > 3:
        parts = sys.argv[3].split('-')
        if len(parts) == 2:
            indices_teste = list(range(int(parts[0]), int(parts[1]) + 1))
        else:
            indices_teste = [int(x) for x in sys.argv[3].split(',')]

    create_representations(csv_descritores, dir_out, test_indices=indices_teste)






