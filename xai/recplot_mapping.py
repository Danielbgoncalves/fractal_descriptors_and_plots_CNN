# model/xai/recplot_mapping.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

TAMANHO_ORIGINAL_DESCRITORES = 180  # Matriz N x N original[cite: 1, 2]
TAMANHO_REDE = 224                  # Entradas da CNN (224x224)[cite: 1]

# Lista exata de raios ímpares de 3 a 41 (total de 20 raios)
RAIOS = list(range(3, 42, 2))


def pixel_to_descriptor(row: int, col: int, n_descritores: int = TAMANHO_ORIGINAL_DESCRITORES, tamanho_final: int = TAMANHO_REDE):
    """
    Mapeia um pixel (row, col) do heatmap 224x224 para o par de descritores originais
    (descritor_row, descritor_col) no espaço 180x180 e decodifica a semântica de cada um.[cite: 1]
    """
    fator_escala = n_descritores / tamanho_final  # 180 / 224 ≈ 0.8035
    
    desc_row = int(row * fator_escala)
    desc_col = int(col * fator_escala)
    
    desc_row = min(max(0, desc_row), n_descritores - 1)
    desc_col = min(max(0, desc_col), n_descritores - 1)
    
    info_row = decodificar_indice_descritor(desc_row)
    info_col = decodificar_indice_descritor(desc_col)
    
    return (desc_row, desc_col), (info_row, info_col)


def decodificar_indice_descritor(idx: int):
    """
    Dado um índice [0..179], retorna a métrica, a função (P, G, H) e o raio r correspondentes.
    """
    # 1. Identifica a Métrica
    if idx < 60:
        metrica = "Minkowski"
        local_idx = idx
    elif idx < 120:
        metrica = "Euclidiana"
        local_idx = idx - 60
    else:
        metrica = "Manhattan"
        local_idx = idx - 120
        
    if local_idx < 20:
        funcao = "p"
        raio_idx = local_idx
    elif local_idx < 40:
        funcao = "g"
        raio_idx = local_idx - 20
    else:
        funcao = "h"
        raio_idx = local_idx - 40
        
    raio = RAIOS[raio_idx]
    
    return {
        "indice": idx,
        "metrica": metrica,
        "funcao": funcao,
        "raio": raio,
        "label": f"{metrica}_{funcao}(r={raio})"
    }


def definir_bandas_metricas():
    """
    Retorna as faixas do RecPlot divididas pelas 3 MÉTRICAS de distância.
    """
    return {
        "Minkowski": (0, 60),
        "Euclidiana": (60, 120),
        "Manhattan": (120, 180)
    }


def definir_bandas_funcoes():
    """
    Retorna as 9 sub-faixas divididas por MÉTRICA + FUNÇÃO (P, G, H).[cite: 2]
    """
    subbandas = {}
    metricas = ["Minkowski", "Euclidiana", "Manhattan"]
    funcoes = ["P", "G", "H"]
    
    idx = 0
    for m in metricas:
        for f in funcoes:
            subbandas[f"{m} ({f})"] = (idx, idx + 20)
            idx += 20
            
    return subbandas


def overlay_bandas_recplot(heatmap: np.ndarray, img_rec_np: np.ndarray, detalhar_funcoes: bool = False, alpha_heatmap: float = 0.5):
    """
    Desenha o RecPlot com o heatmap de atribuição e as grades exatas das métricas (ou funções).[cite: 2]
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    ax.imshow(img_rec_np)
    ax.imshow(heatmap, cmap="jet", alpha=alpha_heatmap)
    
    bandas = definir_bandas_funcoes() if detalhar_funcoes else definir_bandas_metricas()
    fator_conversao = TAMANHO_REDE / TAMANHO_ORIGINAL_DESCRITORES  # 224 / 180
    
    cores = ["#FF3333", "#33FF33", "#3388FF", "#FF9900", "#CC33FF", "#00FFFF", "#FFFF00", "#FF00AA", "#FFFFFF"]
    
    for idx, (nome_banda, (inicio, fim)) in enumerate(bandas.items()):
        p_inicio = inicio * fator_conversao
        p_fim = fim * fator_conversao
        largura = p_fim - p_inicio
        
        cor = cores[idx % len(cores)]
        
        # Quadrado delimitando o bloco na matriz de recorrência
        rect = patches.Rectangle(
            (p_inicio, p_inicio), largura, largura,
            linewidth=1.5, edgecolor=cor, facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        
        # Rotula no mapa
        ax.text(
            p_inicio + 3, p_inicio + (12 if detalhar_funcoes else 18), nome_banda,
            color=cor, fontsize=8 if detalhar_funcoes else 10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.7)
        )
        
    ax.set_title("XAI RecPlot — Mapeamento por Métricas de Distância e Percolação (P, G, H)", fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    
    return fig


def avaliar_importancia_por_metrica_e_funcao(modelo: torch.nn.Module, img_rec_tensor: torch.Tensor, target_class: int):
    """
    Avaliação Quantitativa (Oclusão): Mascara sequencialmente cada bloco (Métrica / Função)
    e mede a queda na confiança do modelo.[cite: 2]
    """
    modelo.eval()
    if img_rec_tensor.dim() == 3:
        img_rec_tensor = img_rec_tensor.unsqueeze(0)
        
    device = next(modelo.parameters()).device
    img_rec_tensor = img_rec_tensor.to(device)
    
    with torch.no_grad():
        out_orig = torch.softmax(modelo(img_rec_tensor), dim=1)
        prob_base = out_orig[0, target_class].item()
        
    subbandas = definir_bandas_funcoes()
    fator_conversao = TAMANHO_REDE / TAMANHO_ORIGINAL_DESCRITORES[cite: 1]
    
    impacto_bandas = {}
    
    for nome_banda, (inicio, fim) in subbandas.items():
        img_oclusa = img_rec_tensor.clone()
        
        p_in = int(inicio * fator_conversao)
        p_out = int(fim * fator_conversao)
        
        # Oclui o bloco correspondente àquela sub-função (p, g ou h para a métrica)[cite: 2]
        img_oclusa[:, :, p_in:p_out, p_in:p_out] = 0.0
        
        with torch.no_grad():
            out_ocluso = torch.softmax(modelo(img_oclusa), dim=1)
            prob_oclusa = out_ocluso[0, target_class].item()
            
        queda_prob = prob_base - prob_oclusa
        impacto_bandas[nome_banda] = queda_prob
        
    return impacto_bandas