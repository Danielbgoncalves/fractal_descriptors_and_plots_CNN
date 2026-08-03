import os
import pandas as pd
from model.dataset import EnsembleTestDataset
from xai.case_selection import (
    casos_corrigidos_pelo_ensemble,
    casos_por_confianca,
    casos_discordancia_entre_branches,
    obter_amostras_do_dataset
)

def executar_teste_etapa6():
    print("=" * 70)
    print("INICIANDO TESTE DA ETAPA 6 — SELEÇÃO DE ESTUDOS DE CASO (predicoes.csv)")
    print("=" * 70)

    csv_path = "resultados_displasia/predicoes.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo {csv_path} não encontrado! Execute a geração de métricas primeiro.")

    print(f"[1/4] Lendo arquivo de predições: {csv_path}...")
    df_pred = pd.read_csv(csv_path)
    seed_teste = 7

    # 1. Teste: Casos onde um modelo errou sozinho mas o ensemble corrigiu
    print(f"\n[2/4] Buscando casos corrigidos pelo Ensemble na Seed {seed_teste}:")
    cenario_ind = "MobileNet_RecPlot + MobileNet_RecPlot"
    cenario_ens = "MobileNet_Original + MobileNet_RecPlot"

    casos_corrigidos = casos_corrigidos_pelo_ensemble(
        df=df_pred,
        seed=seed_teste,
        cenario_individual=cenario_ind,
        cenario_ensemble=cenario_ens
    )
    print(f"  • Cenário: '{cenario_ind}' (Errou) -> '{cenario_ens}' (Acertou)")
    print(f"  • Amostras encontradas (samples): {casos_corrigidos}")

    # 2. Teste: Casos por Nível de Confiança
    print(f"\n[3/4] Buscando casos por confiança (Cenário: '{cenario_ens}'):")
    
    acertos_conf = casos_por_confianca(df_pred, seed=seed_teste, cenario=cenario_ens, tipo="acerto_confiante", n=3)
    erros_conf = casos_por_confianca(df_pred, seed=seed_teste, cenario=cenario_ens, tipo="erro_confiante", n=3)
    fronteira = casos_por_confianca(df_pred, seed=seed_teste, cenario=cenario_ens, tipo="fronteira", n=3)

    print(f"  • Top Acertos Confiantes (samples): {acertos_conf}")
    print(f"  • Top Erros Confiantes   (samples): {erros_conf}")
    print(f"  • Casos de Fronteira    (samples): {fronteira}")

    # 3. Teste: Discordância entre Branches
    print(f"\n[4/4] Buscando discordâncias entre MobileNet_RecPlot e EffNet_RecPlot:")
    casos_disc = casos_discordancia_entre_branches(
        df=df_pred,
        seed=seed_teste,
        branch_a="mobilenet_recplot",
        branch_b="effnet_recplot"
    )
    print(f"  • Amostras em que os dois modelos discordam (samples): {casos_disc}")

    # 4. Validação de Carregamento dos Caminhos dos Arquivos
    print("\n" + "-" * 70)
    print("VALIDANDO MAPEAMENTO PARA OS ARQUIVOS REAIS DO DATASET:")
    print("-" * 70)

    dataset = EnsembleTestDataset(
        "C:\\Users\\IFTM-ITB\\Desktop\\EnsembleFractal\\datasets\\daniel_tentando\\novo\\RPnew\\teste", 
        ['healthy', 'severe'],
        dir_originais="data/displasia/test/originais",
        dir_recplots="data/displasia/test/recplots"
    )

    # Escolhe um conjunto de amostras para testar a recuperação dos arquivos
    amostras_alvo = casos_corrigidos if casos_corrigidos else acertos_conf
    if amostras_alvo:
        detalhes_amostras = obter_amostras_do_dataset(dataset, amostras_alvo[:2])
        for idx, item in zip(amostras_alvo[:2], detalhes_amostras):
            print(f"\n  [Sample #{idx}]")
            print(f"    - Imagem Original: {item['path_orig']}")
            print(f"    - RecPlot PNG:     {item['path_rec']}")
            print(f"    - Label Real:      {item['label']}")

    print("\n" + "=" * 70)
    print("[OK] TESTE DA ETAPA 6 CONCLUÍDO COM SUCESSO!")
    print("=" * 70)

if __name__ == "__main__":
    executar_teste_etapa6()