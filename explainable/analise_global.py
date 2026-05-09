import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importando as funções do seu novo módulo modularizado
from .xai_core import extrac_explanation_2D
from model.model import carregar_modelo
from model.utils import DEVICE

def generate_global_freq_map(model, lista_imgs, target_class, classes, top_percent=15):
    '''
    Varre a list_imgs, filtra pelas com previsão correta e
    cria um Mpa de Frequência mostrando quais pixels são consistentemente importantes.

    top_percent: Define que apenas o X% pixels mais importantes de cada imagem
                vão votar no mapa final
    '''

    total_imgs = len(lista_imgs)
    total_corretas = 0

    # padrão das redes 224,224 nesse trabalho
    soma_votos = np.zeros((224,224), dtype=np.float32)

    print(f"Analisando {total_imgs} imagens para classe alvo: {classes[target_class]}")

    for img_path in lista_imgs:
        _, attr_np, _, info_pred = extrac_explanation_2D(model, img_path, target_class, classes)

        if info_pred['classe_pred'] == info_pred['classe_real']:
            total_corretas += 1

            importancia_2d = np.sum(np.abs(attr_np), axis=2)
            
            threshold = np.percentile(importancia_2d.flatten(), 100 - top_percent)

            mascara_voto = (importancia_2d >= threshold).astype(np.float32)

            soma_votos += mascara_voto
    
    print(f'Concluído! Acertos contados: {total_corretas}/{total_imgs}'
          f'({(total_corretas/total_imgs)*100:.1f}%)')
    
    if total_corretas > 0:
        mapa_frequencia = soma_votos / total_corretas
    else:
        mapa_frequencia = np.zeros_like(soma_votos)

    return mapa_frequencia, total_corretas

def plotar_mapa_global(mapa_frequencia, nome_classe, total_corretas):
    
    plt.figure(figsize=(10,8))
    
    sns.heatmap(mapa_frequencia, cmap='inferno', vmin=0, vmax=np.max(mapa_frequencia),
                cbar_kws={'label': 'Frequncia^de Importância (1.0 = 100%)'})
    
    plt.title(f'Assinatura Global F-RecPlot (Percolação) - Classe: {nome_classe}\n'
              f'Baseado em {total_corretas} predições corretas', fontsize=14, pad=15)
    
    plt.xlabel("Eixo X (Tempo/Atraso t2)")
    plt.ylabel("Eixo Y (Tempo/Atraso t1)")
    
    # Inverte o eixo Y para corresponder às coordenadas de imagem padrão, se necessário
    # plt.gca().invert_yaxis() 
    
    plt.tight_layout()
    plt.savefig(f"mapa_global_{nome_classe}.png", dpi=300)
    plt.show()

def plotar_mapa_contraste(mapa_severe, mapa_healthy):
    # Subtrai um do outro
    mapa_diff = mapa_severe - mapa_healthy
    
    plt.figure(figsize=(10, 8))
    
    # Descobre o valor máximo absoluto para centralizar o zero na cor branca
    limite = np.max(np.abs(mapa_diff))
    
    # 'coolwarm' ou 'RdBu_r' são perfeitos para isso (Azul = Negativo, Vermelho = Positivo)
    sns.heatmap(mapa_diff, cmap='coolwarm', vmin=-limite, vmax=limite, 
                cbar_kws={'label': 'Contraste (Azul = Mais imp. no Healthy | Vermelho = Mais imp. no Severe)'})
    
    plt.title("Mapa de Contraste XAI: Severe vs Healthy\n(O que diferencia as classes?)", fontsize=14, pad=15)
    plt.xlabel("Eixo X (Tempo/Atraso t2)")
    plt.ylabel("Eixo Y (Tempo/Atraso t1)")
    
    plt.tight_layout()
    plt.savefig("mapa_contraste.png", dpi=300)
    plt.show()

if __name__ == '__main__':

    #--- Configurações
    weigths_path = r"models\42\mobilenet_RP_perc.pth" # Esse aqui é do continuous
    backbone = 'mobilenet'
    target_class = 0 #  saudavel = 0, severo = 1
    classes = ["healthy", "severe"]
    num_classes = 2


    # Tomar cuidado com BIN / CONTINUOUS
    pasta_saudavel = r"C:\Users\IFTM-ITB\Desktop\EnsembleFractal\datasets\originais_RPperc_continuous\teste\healthy\RP_perc"
    lista_img_saudavel = glob.glob(os.path.join(pasta_saudavel, "*.png")) # ou .tif para originais

    pasta_severo = r"C:\Users\IFTM-ITB\Desktop\EnsembleFractal\datasets\originais_RPperc_continuous\teste\severe\RP_perc"
    lista_img_severo = glob.glob(os.path.join(pasta_severo, "*.png")) # ou .tif para originais

    #--- Execução
    model = carregar_modelo(backbone, num_classes=2, path_weights=weigths_path)
    model.to(DEVICE)
    model.eval()

    mapa_freq_saudavel, qtd_acertos = generate_global_freq_map(
        model, lista_img_saudavel, target_class=0, classes=classes, top_percent=20
    )

    mapa_freq_severo, qtd_acertos = generate_global_freq_map(
        model, lista_img_severo, target_class=1, classes=classes, top_percent=20
    )

    plotar_mapa_global(mapa_freq_saudavel, "Healthy", qtd_acertos)
    plotar_mapa_global(mapa_freq_severo, "Severe", qtd_acertos)
    plotar_mapa_contraste(mapa_freq_severo, mapa_freq_saudavel)