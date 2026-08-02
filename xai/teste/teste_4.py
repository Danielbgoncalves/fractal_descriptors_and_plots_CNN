import os
import torch
import matplotlib.pyplot as plt

# Importações dos seus módulos
from model.dataset import EnsembleTestDataset
from model.utils import transform_originais, transform_recplot
from xai.loader import carregar_todos_os_branches
from xai.sanity_check import (
    rodar_sanity_check_todos_branches,
    plotar_curva_sanidade
)

def executar_teste_etapa4():
    print("=" * 70)
    print("INICIANDO ETAPA 4 — TESTE DE SANIDADE POR RANDOMIZAÇÃO EM CASCATA")
    print("  (Adebayo et al., 2018 — Sanity Checks for Saliency Maps)")
    print("=" * 70)

    # 1. Configurações base
    seed = 7
    num_classes = 2  # Exemplo Displasia
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = "resultados_displasia/models"
    
    # 2. Carregar os 4 branches treinados da seed
    print(f"\n[1/4] Carregando modelos treinados da seed {seed} no dispositivo '{device}'...")
    branches = carregar_todos_os_branches(
        seed=seed,
        num_classes=num_classes,
        models_dir=models_dir,
        device=device
    )

    # 3. Carregar uma amostra de teste real
    print("\n[2/4] Carregando amostra de teste (Original + RecPlot)...")
    dataset = EnsembleTestDataset(
        'C:\\Users\\IFTM-ITB\\Desktop\\EnsembleFractal\\datasets\\daniel_tentando\\novo\\RPnew\\teste',
        ['healthy', 'severe'], 
        transform_original=transform_originais, 
        transform_recplot=transform_recplot
    )

    # Pega uma amostra de teste (Original + RecPlot)
    img_orig_tensor, img_rec_tensor, label = dataset[0]
    label_int = label.item() if isinstance(label, torch.Tensor) else label

    print(f"      - Amostra carregada com sucesso! Classe Alvo (Label): {label_int}")

    # 4. Rodar o Teste em Cascata para os 4 Branches (usando Grad-CAM)
    print("\n[3/4] Rodando randomização em cascata para os 4 branches via Grad-CAM...")
    print("      (Aguarde alguns segundos enquanto a rede é re-inicializada bloco a bloco...)")
    
    resultados = rodar_sanity_check_todos_branches(
        branches=branches,
        img_orig_tensor=img_orig_tensor,
        img_rec_tensor=img_rec_tensor,
        target_class=label_int,
        metodo="gradcam"
    )

    # Exibe tabela formatada de resultados no terminal
    print("\n" + "=" * 70)
    print("RESULTADOS DO TESTE DE SANIDADE (Correlação de Spearman vs. Modelo Treinado):")
    print("=" * 70)
    
    for nome_branch, res_list in resultados.items():
        print(f"\n>>> Branch: {nome_branch}")
        print(f"  {'Bloco Randomizado':<25} | {'Spearman':<10} | {'SSIM':<10}")
        print("  " + "-" * 51)
        for r in res_list:
            bloco_nome = r['bloco']
            sp_val = r['spearman']
            ssim_val = r['ssim']
            print(f"  {bloco_nome:<25} | {sp_val:<10.4f} | {ssim_val:<10.4f}")

    # 5. Gerar e salvar as figuras de diagnóstico
    print("\n[4/4] Gerando gráficos de diagnóstico...")
    os.makedirs("xai_figuras", exist_ok=True)

    # Gráfico de Spearman
    fig_spearman = plotar_curva_sanidade(
        resultados, 
        metrica="spearman", 
        titulo=f"Sanity Check (Grad-CAM) — Spearman (Seed {seed})"
    )
    caminho_sp = "xai_figuras/sanity_check_spearman.png"
    fig_spearman.savefig(caminho_sp, dpi=300)
    plt.close(fig_spearman)

    # Gráfico de SSIM
    fig_ssim = plotar_curva_sanidade(
        resultados, 
        metrica="ssim", 
        titulo=f"Sanity Check (Grad-CAM) — SSIM (Seed {seed})"
    )
    caminho_ssim = "xai_figuras/sanity_check_ssim.png"
    fig_ssim.savefig(caminho_ssim, dpi=300)
    plt.close(fig_ssim)

    print(f"\n[OK] Gráfico de Spearman salvo em: {caminho_sp}")
    print(f"[OK] Gráfico de SSIM salvo em:     {caminho_ssim}")
    print("=" * 70)

if __name__ == "__main__":
    executar_teste_etapa4()