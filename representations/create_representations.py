from skimage.transform import resize
import matplotlib.pyplot as plt
import imageio.v3 as iio
import pandas as pd
import numpy as np 
import os



from .util import organize_by_nature
from .RP import generate_RP
#from .GAF import generate_GAF
#from .MTF import generate_MTF

REPRESENTATIONS_MAP = {
     'RP': generate_RP,
    #  'GAF': generate_GAF,
    #  'MTF': generate_MTF
}

def salve_images(images, names, subfolders, base_out_path, technique):

    unique_subfolders = np.unique(subfolders)

    for subfolder in unique_subfolders:
        dir_path = os.path.join(base_out_path, technique, subfolder)
        os.makedirs(dir_path, exist_ok=True)
    
    for i in range(images.shape[0]):

        img_224 = resize(images[i], (224, 224), anti_aliasing=True)
        img_uint8 = ( img_224 * 255 ).astype(np.uint8)

        nome_original = names[i]
        nome_sem_extensao = os.path.splitext(nome_original)[0]
        novo_nome = f'{nome_sem_extensao}.png'

        path = os.path.join(base_out_path, technique, subfolders[i], novo_nome)
        iio.imwrite(path , img_uint8)

def create_representations(features_path, out_path, plots=['RP']):

    features_df = pd.read_csv(features_path, sep=';')
    print(f'O csv tem shape: {features_df.shape}')

    if features_df.shape[0] < 1: 
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

    for plot_type in plots:
        plot_type.upper()

        if plot_type not in REPRESENTATIONS_MAP:
            print(f'Erro: técnica {plot_type} não implementada. Ignorando...')
            continue

        generator_func = REPRESENTATIONS_MAP[plot_type]
        input_data = data_map[plot_type]

        generated_imgs = generator_func(input_data)

        salve_images(
            images=generated_imgs,
            names=imgs_names,
            subfolders=imgs_subfolders,
            base_out_path=out_path,
            technique=plot_type
        )

    print(f'Salvo com sucesso em {out_path}')


csv_descritores = r"representations\teste.csv"
create_representations(csv_descritores, "saida3")

    





