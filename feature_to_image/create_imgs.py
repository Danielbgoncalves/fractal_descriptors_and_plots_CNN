import pandas as pd
import numpy as np
import time
import os

from reshapeRecPlot import reshapeRecPlot
from reshapeClassical import reshapeClassical


def create_imgs(origem_csv, destino_base, mode='both'):
    '''
    Lê um CSV com as colunas de features e opcionalmente `relative_path` e `image_name`,
    e salva as imagens diretamente em `destino_base` dentro das pastas
    `F-RecPlot` e `F-Classical`, preservando a hierarquia de `relative_path`.
    '''
    tic = time.time()

    df = pd.read_csv(origem_csv, sep=',')
    rel_paths = None
    image_names = None
    if 'relative_path' in df.columns and 'image_name' in df.columns:
        rel_paths = df['relative_path'].astype(str).tolist()
        image_names = df['image_name'].astype(str).tolist()

    feature_cols = [c for c in df.columns if c not in ('relative_path', 'image_name')]
    features = df[feature_cols].to_numpy(dtype=np.float64)

    rec_out_paths = None
    class_out_paths = None
    if rel_paths is not None and image_names is not None:
        rec_out_paths = []
        class_out_paths = []
        for rel, name in zip(rel_paths, image_names):
            base, _ = os.path.splitext(name)
            rel_dir = os.path.dirname(rel)
            out_rel = os.path.join(rel_dir, base + '.png') if rel_dir else (base + '.png')
            rec_out_paths.append(os.path.join(destino_base, 'F-RecPlot', out_rel).replace('\\', '/'))
            class_out_paths.append(os.path.join(destino_base, 'F-Classical', out_rel).replace('\\', '/'))

    # chamar as funções conforme o modo
    if mode in ('both', 'recplot'):
        reshapeRecPlot(destino_base, features, out_paths=rec_out_paths)
    if mode in ('both', 'classical'):
        reshapeClassical(destino_base, features, out_paths=class_out_paths)

    toc = time.time()
    intervalo = toc - tic
    print(f'Imagens salvas em {destino_base}')
    print(f"Concluído em {intervalo:.3f} segundos")


if __name__ == "__main__":
    import sys
    origem = sys.argv[1]
    destino = sys.argv[2]
    mode = sys.argv[3].lower() if len(sys.argv) > 3 else 'both'
    create_imgs(origem, destino, mode=mode)