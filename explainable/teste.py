import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def organize_by_nature(features):
    # Sua função original
    perc = np.hstack((features[:, 0:60], features[:, 100:160], features[:, 200:260]))
    lac = np.hstack((features[:, 60:80 ], features[:, 160:180 ], features[:, 260:280 ]))
    massa = np.hstack((features[:, 80:100], features[:, 180:200], features[:, 280:300]))
    globais = features[:, 300:363]
    global_df = np.hstack((massa, globais))
    return perc, lac, global_df   

def analisar_percolacao(caminho_healthy, caminho_severe):
    # 1. Carregar os dados
    # IMPORTANTE: Se o seu CSV tiver cabeçalho ou coluna de "classe", 
    # você precisará remover essa coluna antes de jogar no organize_by_nature.
    # Exemplo: df_healthy = pd.read_csv(...).drop(columns=['classe']).values
    
    features_healthy = pd.read_csv(caminho_healthy).values
    features_severe = pd.read_csv(caminho_severe).values
    
    # 2. Extrair apenas a Percolação (180 features)
    perc_healthy, _, _ = organize_by_nature(features_healthy)
    perc_severe, _, _ = organize_by_nature(features_severe)
    
    # 3. Cálculos Estatísticos (Eixo 0 = média de todas as amostras para cada feature)
    media_h = np.mean(perc_healthy, axis=0)
    std_h = np.std(perc_healthy, axis=0)
    
    media_s = np.mean(perc_severe, axis=0)
    std_s = np.std(perc_severe, axis=0)
    
    eixo_x = np.arange(180) # Nossas 180 features

    # ==========================================
    # PLOT 1: Assinatura do Sinal Médio
    # ==========================================
    plt.figure(figsize=(15, 6))
    
    # Plot Healthy (Verde)
    plt.plot(eixo_x, media_h, label="Média Healthy", color="green", linewidth=2)
    plt.fill_between(eixo_x, media_h - std_h, media_h + std_h, color="green", alpha=0.2)
    
    # Plot Severe (Vermelho)
    plt.plot(eixo_x, media_s, label="Média Severe", color="red", linewidth=2, linestyle="--")
    plt.fill_between(eixo_x, media_s - std_s, media_s + std_s, color="red", alpha=0.2)
    
    # Linhas verticais para separar as distâncias (Mink, Eucl, Manh)
    plt.axvline(60, color='grey', linestyle=':', alpha=0.7)
    plt.axvline(120, color='grey', linestyle=':', alpha=0.7)
    
    plt.title("Assinatura Média da Percolação (180 Features)")
    plt.xlabel("Índice da Feature (0-59: Mink | 60-119: Eucl | 120-179: Manh)")
    plt.ylabel("Valor Numérico (Normalizado)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # PLOT 2: Análise de Separação com PCA (2D)
    # ==========================================
    # Juntar os dados para o PCA ter a mesma base de transformação
    X_total = np.vstack((perc_healthy, perc_severe))
    
    # Criar labels para pintar o gráfico (0 = healthy, 1 = severe)
    y_total = np.array([0]*perc_healthy.shape[0] + [1]*perc_severe.shape[0])
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_total)
    
    # Variância explicada (o quanto esses 2 eixos resumem da informação total)
    var_exp = pca.explained_variance_ratio_ * 100
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1], 
        hue=y_total, palette={0: "green", 1: "red"}, 
        alpha=0.6, s=50
    )
    plt.title(f"Separação das Classes (PCA)\nExplica {var_exp[0]:.1f}% + {var_exp[1]:.1f}% da variação")
    plt.xlabel(f"Componente Principal 1 ({var_exp[0]:.1f}%)")
    plt.ylabel(f"Componente Principal 2 ({var_exp[1]:.1f}%)")
    
    # Ajustando a legenda
    plt.legend(title="Classes", labels=["Healthy", "Severe"])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Exemplo de uso:
# analisar_percolacao("features_healthy.csv", "features_severe.csv")