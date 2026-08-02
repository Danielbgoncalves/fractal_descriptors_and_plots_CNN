import os
import torch
import matplotlib.pyplot as plt

# Importações dos módulos do seu projeto
from model.utils import transform_originais, transform_recplot
from model.dataset import EnsembleTestDataset
from xai.loader import carregar_todos_os_branches, obter_camada_alvo
from xai.attributions import gerar_atribuicao, desnormalizar
from xai.recplot_mapping import (
    pixel_to_descriptor,
    overlay_bandas_recplot,
    avaliar_importancia_por_metrica_e_funcao
)

def executar_teste_etapa3():
    print("=" * 60)
    print("INICIANDO TESTE DA ETAPA 3 — MAPEAMENTO DE DESCRITORES FRACTAIS")
    print("=" * 60)
    
    # Configurações
    seed = 7
    num_classes = 2  # Exemplo Displasia
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Carrega os modelos da seed
    print(f"\n[1/5] Carregando branches da seed {seed} (Device: {device})...")
    branches = carregar_todos_os_branches(
        seed=seed, 
        num_classes=num_classes, 
        models_dir="resultados_displasia/models", 
        device=device
    )
    modelo_rec = branches["mobilenet_recplot"]
    camada = obter_camada_alvo(modelo_rec)
    
    # 2. Carrega uma amostra do dataset
    print("[2/5] Carregando dataset de teste...")
    dataset = EnsembleTestDataset(
        "C:\\Users\\IFTM-ITB\\Desktop\\EnsembleFractal\\datasets\\daniel_tentando\\novo\\RPnew\\teste",
        ['healthy', 'severe'], 
        transform_original=transform_originais, 
        transform_recplot=transform_recplot
    )

    _, img_rec, label = dataset[0]
    label_int = label.item() if isinstance(label, torch.Tensor) else label
    
    # 3. Teste de Atribuição (Grad-CAM)
    print("[3/5] Gerando mapa de atribuição (Grad-CAM)...")
    mapa_gradcam = gerar_atribuicao(
        modelo=modelo_rec,
        input_tensor=img_rec,
        target_class=label_int,
        camada_alvo=camada,
        metodo="gradcam"
    )
    
    # Normalização Min-Max do heatmap para [0, 1]
    heatmap = mapa_gradcam.squeeze().detach().cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # 4. Teste de Decodificação de Pixels do Heatmap (Pixel -> Descritor Físico)
    print("\n[4/5] Testando decodificação de pontos de maior relevância no heatmap:")
    # Pega as 3 coordenadas (row, col) com maior ativação no Grad-CAM
    indices_top = torch.topk(torch.tensor(heatmap).flatten(), k=3).indices
    
    for rank, idx in enumerate(indices_top, 1):
        row = (idx // 224).item()
        col = (idx % 224).item()
        
        (d_row, d_col), (info_row, info_col) = pixel_to_descriptor(row, col)
        val_ativacao = heatmap[row, col]
        
        print(f"\n  • Top #{rank} Ativação (Pixel [{row}, {col}] = {val_ativacao:.3f}):")
        print(f"    - Linha {d_row}: {info_row['label']}")
        print(f"    - Coluna {d_col}: {info_col['label']}")
        print(f"    - Par de Interação: {info_row['metrica']} ({info_row['funcao']}, r={info_row['raio']}) x {info_col['metrica']} ({info_col['funcao']}, r={info_col['raio']})")

    # 5. Avaliação Quantitativa de Oclusão por Sub-Banda (P, G, H x Métricas)
    print("\n[5/5] Calculando impacto de oclusão por bloco (Métrica + Função P,G,H):")
    impacto = avaliar_importancia_por_metrica_e_funcao(
        modelo=modelo_rec, 
        img_rec_tensor=img_rec, 
        target_class=label_int
    )
    
    print("\nQueda de probabilidade da classe correta ao ocluir cada região:")
    for bloco, queda in impacto.items():
        barra = "█" * int(max(0, queda) * 50)
        print(f"  - {bloco:<20}: {queda:+.4f} | {barra}")

    # 6. Salva Figura de Exemplo
    img_rec_np = desnormalizar(img_rec, tipo="rec")
    fig = overlay_bandas_recplot(heatmap, img_rec_np, detalhar_funcoes=True, alpha_heatmap=0.4)
    
    out_fig = "teste_etapa3_resultado.png"
    fig.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[OK] Figura gerada com sucesso e salva em: {out_fig}")
    print("=" * 60)

if __name__ == "__main__":
    executar_teste_etapa3()