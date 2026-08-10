# '''
# run_analise_completa.py

# Orquestrador único da Etapa "análise XAI completa". Não reimplementa
# nada — só chama, na ordem certa, o que já existe em xai/*.py e salva
# tudo de forma organizada em analise_xai/.

# IMPORTANTE — aplique antes de rodar:
#   1. xai/xai_metrics.py: normalise=True no quantus.MaxSensitivity
#   2. xai/comparacao_arquiteturas.py: `else img_rec` (linha do bug do
#      representação "recplot" usando img_orig)
#   3. xai/comparacao_datasets.py: "Queda_prob" -> "queda_prob"
#   4. xai/comparacao_datasets.py: figzise -> figsize
# (ver mensagem anexa a este script para o motivo de cada patch)

# Como funciona:
#   - cria TODAS as subpastas de analise_xai/ antes de chamar qualquer
#     função (pedido explícito: "crie as subpastas de antemão")
#   - cada etapa é isolada numa função e passa por `rodar_etapa()`, que:
#       - pula a etapa se já existe um marcador `.OK` (retomável — se o
#         processo cair no meio, rodar de novo não repete o que já
#         passou)
#       - registra início/fim/erro num log único, sem travar as etapas
#         seguintes se uma falhar (é isso que "quero que seja contínuo"
#         significa aqui: o script não para no primeiro erro)

# PREENCHA antes de rodar:
#   - DATASETS[i]["test_dir"]: caminho real do dataset de teste (os
#     scripts xai/teste/*.py atuais têm isso hardcoded pra uma máquina
#     específica — não posso adivinhar o seu caminho aqui)

# Parâmetros que você pediu pra revisar:
#   - N_CONFIANTES = 8 (era 15 no teste_10.py original — 15 sobre 46
#     amostras de displasia já cobria ~1/3 do teste; 8 é mais seletivo)
# '''
# import os
# import time
# import traceback
# from datetime import datetime

# import torch
# import pandas as pd

# from model.utils import SEEDS, transform_originais, transform_recplot
# from model.dataset import EnsembleTestDataset

# from xai.loader import carregar_todos_os_branches, obter_camada_alvo, BRANCH_FILES
# from xai.attributions import gerar_atribuicao, desnormalizar
# from xai.case_selection import BRANCH_PARA_CENARIO_INDIVIDUAL, casos_por_confianca, casos_por_margem
# from xai.seed_aggregation import (
#     mapa_agregado_para_amostra, figura_mapa_agregado, salvar_mapas_agregados, seed_mais_representativa,
# )
# from xai.threshold_analysis import analisar_limiar_de_decisao, gerar_figura_exemplos, _heatmap_unico
# from xai.xai_metrics import xai_metrics_to_csv
# from xai.comparacao_arquiteturas import (
#     comparar_arquiteturas_em_lotes, cruzar_com_predicoes, resumo_concordancia_vs_ensemble,
# )
# from xai.comparacao_datasets import (
#     importancia_por_banda_dataset, resumo_importancia_por_banda,
#     comparar_importancia_entre_datasets, figura_comparacao_bandas,
# )
# from xai.sanity_check import rodar_sanity_check_todos_branches, plotar_curva_sanidade

# # ============================================================
# # CONFIGURAÇÃO — revise antes de rodar
# # ============================================================

# RAIZ_SAIDA = "analise_xai"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# METODOS = ["gradcam", "ig", "occlusion"]
# METODOS_SANITY_CHECK = ["gradcam", "ig", "occlusion"]   # pedido explícito: "de todos eles"
# METODOS_SEED_AGGREGATION = ["gradcam"]                   # ig/occlusion x 8 seeds x N amostras é MUITO caro; amplie se topar esperar mais

# MARGEM_MAX = 0.1
# N_CONFIANTES = 8          # reduzido de 15, a seu pedido
# N_AMOSTRAS_XAI_METRICS = None   # None = usa todas as amostras de teste do dataset
# N_AMOSTRAS_COMPARACAO_ARQ = 20  # subset por custo (8 seeds x N amostras x 2 arquiteturas)
# N_AMOSTRAS_COMPARACAO_DATASETS = 6  # 8 seeds x 9 bandas x N amostras x 2 branches — é o mais caro de todos
# N_AMOSTRAS_SEED_AGG_POR_CATEGORIA = 2  # quantas amostras de cada categoria (acerto/fronteira/erro) por branch

# SEED_FIXA_EXEMPLOS = SEEDS[0]  # mesma convenção de xai/teste/teste_3.py e teste_4.py (seed=7)
# SAMPLE_FIXA_EXEMPLO = 0        # amostra única usada nas figuras de heatmap/sanity check de exemplo

# DATASETS = [
#     dict(
#         nome="displasia",
#         classes=["healthy", "severe"],
#         num_classes=2,
#         test_dir=r"C:\\Users\\IFTM-ITB\\Desktop\\EnsembleFractal\\datasets\\daniel_tentando\\novo\\RPnew\\teste",   # <-- preencher
#         models_dir="resultados_displasia/models",
#         predicoes_csv="resultados_displasia/predicoes.csv",
#         resultados_testes_csv="resultados_displasia/resultados_testes.csv",
#     ),
#     dict(
#         nome="pulmao",
#         classes=["aca_md", "nor", "scc_md"],
#         num_classes=3,
#         test_dir=r"C:\\Users\\IFTM-ITB\\Desktop\\EnsembleFractal\\datasets\\daniel_tentando\\dataset_lung\\teste",   # <-- preencher
#         models_dir="resultados_pulmao/models",
#         predicoes_csv="resultados_pulmao/predicoes.csv",
#         resultados_testes_csv="resultados_pulmao/resultados_testes.csv",
#     ),
# ]

# # combinação de ensemble que isola a arquitetura em cada representação
# # (mesma representação nos dois lados, arquitetura diferente)
# CENARIO_ENSEMBLE_POR_REPRESENTACAO = {
#     "orig": "MobileNet_Original + EffNet_Original",
#     "recplot": "MobileNet_RecPlot + EffNet_RecPlot",
# }


# # ============================================================
# # INFRAESTRUTURA: pastas, log, checkpoint
# # ============================================================

# def caminho_log():
#     return os.path.join(RAIZ_SAIDA, "logs", "execucao.log")


# def log(msg):
#     linha = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
#     print(linha)
#     with open(caminho_log(), "a", encoding="utf-8") as f:
#         f.write(linha + "\n")


# def criar_estrutura_pastas():
#     '''
#     Cria TODAS as subpastas de analise_xai/ de antemão, antes de
#     qualquer função ser chamada — pedido explícito.
#     '''
#     subpastas_por_dataset = [
#         "heatmaps_exemplo",
#         "xai_metricas",
#         "limiar",
#         "seed_aggregation",
#         "comparacao_arquiteturas",
#         "sanity_check",
#     ]

#     os.makedirs(os.path.join(RAIZ_SAIDA, "logs"), exist_ok=True)
#     os.makedirs(os.path.join(RAIZ_SAIDA, "_checkpoints"), exist_ok=True)
#     os.makedirs(os.path.join(RAIZ_SAIDA, "comparacao_datasets"), exist_ok=True)

#     for cfg in DATASETS:
#         nome = cfg["nome"]
#         for sub in subpastas_por_dataset:
#             for metodo in METODOS:
#                 os.makedirs(os.path.join(RAIZ_SAIDA, nome, sub, metodo), exist_ok=True)

#     log(f"Estrutura de pastas criada em {RAIZ_SAIDA}/")


# def rodar_etapa(nome_etapa, func):
#     '''
#     Roda uma etapa com checkpoint: se já rodou (marcador .OK existe),
#     pula. Se der erro, registra e CONTINUA pra próxima etapa — é isso
#     que torna a execução "contínua" mesmo em uma etapa falhando.
#     '''
#     marcador = os.path.join(RAIZ_SAIDA, "_checkpoints", f"{nome_etapa}.OK")

#     if os.path.exists(marcador):
#         log(f"[PULANDO] {nome_etapa} (já concluída — apague o marcador em _checkpoints/ pra refazer)")
#         return

#     log(f"[INICIO] {nome_etapa}")
#     inicio = time.time()
#     try:
#         func()
#         with open(marcador, "w") as f:
#             f.write(f"concluída em {datetime.now().isoformat()}\n")
#         log(f"[OK] {nome_etapa} — {time.time() - inicio:.1f}s")
#     except Exception as e:
#         log(f"[ERRO] {nome_etapa}: {repr(e)}")
#         log(traceback.format_exc())


# # ============================================================
# # ETAPAS
# # ============================================================

# def carregar_dataset(cfg):
#     return EnsembleTestDataset(
#         cfg["test_dir"], cfg["classes"],
#         transform_original=transform_originais, transform_recplot=transform_recplot,
#     )


# def etapa_heatmaps_exemplo(cfg, dataset, metodo):
#     '''
#     Uma figura simples por método: os 4 branches lado a lado pra UMA
#     amostra fixa (mesma convenção de teste_3.py) — inspeção visual
#     rápida antes de qualquer análise pesada.
#     '''
#     import matplotlib.pyplot as plt

#     branches = carregar_todos_os_branches(SEED_FIXA_EXEMPLOS, cfg["num_classes"], models_dir=cfg["models_dir"], device=DEVICE)
#     img_orig, img_rec, label = dataset[SAMPLE_FIXA_EXEMPLO]
#     target_class = int(label.item()) if isinstance(label, torch.Tensor) else int(label)

#     fig, axes = plt.subplots(1, 4, figsize=(16, 4))
#     for ax, (nome_branch, modelo) in zip(axes, branches.items()):
#         if modelo is None:
#             ax.axis("off")
#             continue
#         tipo_dataset = BRANCH_FILES[nome_branch][1]
#         input_tensor = img_orig if tipo_dataset == "originais" else img_rec
#         mapa = _heatmap_unico(modelo, input_tensor, target_class, metodo=metodo)
#         fundo = desnormalizar(input_tensor, tipo="orig" if tipo_dataset == "originais" else "rec")
#         ax.imshow(fundo)
#         ax.imshow(mapa, cmap="jet", alpha=0.5)
#         ax.set_title(nome_branch, fontsize=10)
#         ax.axis("off")

#     fig.suptitle(f"{cfg['nome']} — seed {SEED_FIXA_EXEMPLOS} — amostra {SAMPLE_FIXA_EXEMPLO} — {metodo}")
#     plt.tight_layout()
#     caminho = os.path.join(RAIZ_SAIDA, cfg["nome"], "heatmaps_exemplo", metodo, "exemplo_4_branches.png")
#     fig.savefig(caminho, dpi=150)
#     plt.close(fig)


# def etapa_xai_metrics(cfg, dataset, metodo):
#     out_dir = os.path.join(RAIZ_SAIDA, cfg["nome"], "xai_metricas", metodo)
#     xai_metrics_to_csv(
#         seeds=SEEDS, num_classes=cfg["num_classes"], models_dir=cfg["models_dir"],
#         dataset=dataset, output_dir=out_dir, metodo=metodo,
#         n_amostras=N_AMOSTRAS_XAI_METRICS, device=DEVICE,
#     )


# def etapa_limiar(cfg, dataset, metodo):
#     out_dir = os.path.join(RAIZ_SAIDA, cfg["nome"], "limiar", metodo)
#     df_resultado, df_resumo = analisar_limiar_de_decisao(
#         cfg["predicoes_csv"], dataset, cfg["nome"], cfg["num_classes"], out_dir,
#         seeds=SEEDS, margem_max=MARGEM_MAX, n_confiantes=N_CONFIANTES,
#         models_dir=cfg["models_dir"], metodo=metodo, device=DEVICE,
#     )
#     for nome_branch in BRANCH_FILES:
#         try:
#             fig = gerar_figura_exemplos(
#                 dataset, df_resultado, nome_branch, SEED_FIXA_EXEMPLOS,
#                 metodo=metodo, models_dir=cfg["models_dir"], num_classes=cfg["num_classes"], device=DEVICE,
#             )
#             fig.savefig(os.path.join(out_dir, f"exemplos_{nome_branch}.png"), dpi=150)
#         except Exception:
#             log(f"    [AVISO] figura de exemplos falhou para {nome_branch}/{metodo}: {traceback.format_exc(limit=1)}")


# def etapa_seed_aggregation(cfg, dataset, metodo):
#     '''
#     Pega algumas amostras de cada categoria (acerto_confiante, fronteira,
#     erro_confiante) — reaproveitando case_selection.py — e gera o mapa
#     agregado entre as 8 seeds pra cada uma, por branch.
#     '''
#     df_pred = pd.read_csv(cfg["predicoes_csv"])
#     out_dir = os.path.join(RAIZ_SAIDA, cfg["nome"], "seed_aggregation", metodo)

#     for nome_branch, cenario in BRANCH_PARA_CENARIO_INDIVIDUAL.items():
#         samples_acerto = casos_por_confianca(df_pred, SEED_FIXA_EXEMPLOS, cenario, tipo="acerto_confiante", n=N_AMOSTRAS_SEED_AGG_POR_CATEGORIA)
#         samples_erro = casos_por_confianca(df_pred, SEED_FIXA_EXEMPLOS, cenario, tipo="erro_confiante", n=N_AMOSTRAS_SEED_AGG_POR_CATEGORIA)
#         df_fronteira = casos_por_margem(df_pred, SEED_FIXA_EXEMPLOS, cenario, margem_max=MARGEM_MAX)
#         samples_fronteira = df_fronteira["sample"].head(N_AMOSTRAS_SEED_AGG_POR_CATEGORIA).tolist()

#         amostras_marcadas = (
#             [(s, "acerto_confiante") for s in samples_acerto]
#             + [(s, "erro_confiante") for s in samples_erro]
#             + [(s, "fronteira") for s in samples_fronteira]
#         )

#         for sample, categoria in amostras_marcadas:
#             linha = df_pred[(df_pred["seed"] == SEED_FIXA_EXEMPLOS) & (df_pred["cenario"] == cenario) & (df_pred["sample"] == sample)]
#             if linha.empty:
#                 continue
#             target_class = int(linha.iloc[0]["y_pred"])

#             img_orig, img_rec, label = dataset[sample]
#             media, desvio, seeds_usadas = mapa_agregado_para_amostra(
#                 nome_branch, img_orig, img_rec, target_class, cfg["num_classes"],
#                 seeds=SEEDS, models_dir=cfg["models_dir"], metodo=metodo, device=DEVICE,
#             )
#             nome_base = f"{nome_branch}_{categoria}_amostra{sample}"
#             salvar_mapas_agregados(media, desvio, seeds_usadas, out_dir, nome_base)

#             tipo_dataset = BRANCH_FILES[nome_branch][1]
#             fundo = desnormalizar(img_orig if tipo_dataset == "originais" else img_rec, tipo="orig" if tipo_dataset == "originais" else "rec")
#             fig = figura_mapa_agregado(media, desvio, img_fundo=fundo, titulo=f"{nome_branch} — {categoria} — amostra {sample}")
#             fig.savefig(os.path.join(out_dir, f"{nome_base}.png"), dpi=150)
#             import matplotlib.pyplot as plt
#             plt.close(fig)


# def etapa_comparacao_arquiteturas(cfg, dataset, metodo):
#     '''
#     Requer o patch do bug `else img_orig` -> `else img_rec` (ver início
#     deste arquivo) — sem o patch, a metade "recplot" desta etapa
#     produz números sem sentido (compara o mesmo input errado nos dois
#     branches recplot).
#     '''
#     df_pred = pd.read_csv(cfg["predicoes_csv"])
#     out_dir = os.path.join(RAIZ_SAIDA, cfg["nome"], "comparacao_arquiteturas", metodo)
#     n_disponivel = min(N_AMOSTRAS_COMPARACAO_ARQ, len(dataset))
#     indices = list(range(n_disponivel))

#     for representacao, cenario_ensemble in CENARIO_ENSEMBLE_POR_REPRESENTACAO.items():
#         partes = []
#         for seed in SEEDS:
#             branches = carregar_todos_os_branches(seed, cfg["num_classes"], models_dir=cfg["models_dir"], device=DEVICE)
#             df_comp = comparar_arquiteturas_em_lotes(branches, dataset, indices, representacao, metodo=metodo)
#             df_comp["seed"] = seed
#             df_cruzado = cruzar_com_predicoes(df_comp, df_pred, seed, cenario_ensemble)
#             partes.append(df_cruzado)

#         df_total = pd.concat(partes, ignore_index=True)
#         df_total.to_csv(os.path.join(out_dir, f"comparacao_{representacao}.csv"), index=False)

#         df_resumo = resumo_concordancia_vs_ensemble(df_total)
#         df_resumo.to_csv(os.path.join(out_dir, f"resumo_{representacao}.csv"))


# def etapa_sanity_check(cfg, dataset, metodo):
#     branches = carregar_todos_os_branches(SEED_FIXA_EXEMPLOS, cfg["num_classes"], models_dir=cfg["models_dir"], device=DEVICE)
#     img_orig, img_rec, label = dataset[SAMPLE_FIXA_EXEMPLO]
#     target_class = int(label.item()) if isinstance(label, torch.Tensor) else int(label)

#     resultados = rodar_sanity_check_todos_branches(branches, img_orig, img_rec, target_class, metodo=metodo)

#     out_dir = os.path.join(RAIZ_SAIDA, cfg["nome"], "sanity_check", metodo)

#     linhas = []
#     for nome_branch, lista in resultados.items():
#         for r in lista:
#             linhas.append({"branch": nome_branch, **r})
#     pd.DataFrame(linhas).to_csv(os.path.join(out_dir, "sanity_check.csv"), index=False)

#     import matplotlib.pyplot as plt
#     for metrica in ("spearman", "ssim"):
#         fig = plotar_curva_sanidade(resultados, metrica=metrica, titulo=f"{cfg['nome']} — sanity check ({metodo}) — {metrica}")
#         fig.savefig(os.path.join(out_dir, f"curva_{metrica}.png"), dpi=150)
#         plt.close(fig)


# def etapa_comparacao_datasets(datasets_cfg, datasets_obj):
#     '''
#     Cross-dataset: só faz sentido nos branches RecPlot (é onde bandas
#     de descritor existem). Requer os 2 patches de comparacao_datasets.py
#     (Queda_prob -> queda_prob, figzise -> figsize).
#     '''
#     out_dir = os.path.join(RAIZ_SAIDA, "comparacao_datasets")
#     resumos = {}

#     for nome_branch in ["mobilenet_recplot", "effnet_recplot"]:
#         resumos[nome_branch] = {}
#         for cfg, dataset in zip(datasets_cfg, datasets_obj):
#             n_disponivel = min(N_AMOSTRAS_COMPARACAO_DATASETS, len(dataset))
#             df_importancia = importancia_por_banda_dataset(
#                 nome_branch, dataset, list(range(n_disponivel)), cfg["num_classes"],
#                 seeds=SEEDS, models_dir=cfg["models_dir"], device=DEVICE,
#             )
#             df_importancia.to_csv(os.path.join(out_dir, f"importancia_{nome_branch}_{cfg['nome']}.csv"), index=False)

#             df_resumo = resumo_importancia_por_banda(df_importancia)
#             df_resumo.to_csv(os.path.join(out_dir, f"resumo_{nome_branch}_{cfg['nome']}.csv"))
#             resumos[nome_branch][cfg["nome"]] = df_resumo

#         nome_a, nome_b = datasets_cfg[0]["nome"], datasets_cfg[1]["nome"]
#         tabela = comparar_importancia_entre_datasets(
#             resumos[nome_branch][nome_a], resumos[nome_branch][nome_b], nome_a=nome_a, nome_b=nome_b,
#         )
#         tabela.to_csv(os.path.join(out_dir, f"comparacao_{nome_branch}_{nome_a}_vs_{nome_b}.csv"))

#         fig = figura_comparacao_bandas(tabela, nome_a, nome_b)
#         fig.savefig(os.path.join(out_dir, f"figura_{nome_branch}_{nome_a}_vs_{nome_b}.png"), dpi=150)
#         import matplotlib.pyplot as plt
#         plt.close(fig)


# # ============================================================
# # MAIN
# # ============================================================

# def main():
#     criar_estrutura_pastas()
#     log("=" * 70)
#     log("INICIANDO ANÁLISE XAI COMPLETA")
#     log("=" * 70)

#     datasets_obj = []
#     for cfg in DATASETS:
#         if "PREENCHER" in cfg["test_dir"]:
#             log(f"[ERRO FATAL] test_dir de {cfg['nome']} não foi preenchido — edite DATASETS no topo do script.")
#             return
#         datasets_obj.append(carregar_dataset(cfg))

#     for cfg, dataset in zip(DATASETS, datasets_obj):
#         nome = cfg["nome"]
#         log(f"--- Dataset: {nome} ({len(dataset)} amostras de teste) ---")

#         for metodo in METODOS:
#             rodar_etapa(f"{nome}__heatmaps_exemplo__{metodo}", lambda cfg=cfg, dataset=dataset, metodo=metodo: etapa_heatmaps_exemplo(cfg, dataset, metodo))

#         for metodo in METODOS:
#             rodar_etapa(f"{nome}__xai_metrics__{metodo}", lambda cfg=cfg, dataset=dataset, metodo=metodo: etapa_xai_metrics(cfg, dataset, metodo))

#         for metodo in METODOS:
#             rodar_etapa(f"{nome}__limiar__{metodo}", lambda cfg=cfg, dataset=dataset, metodo=metodo: etapa_limiar(cfg, dataset, metodo))

#         for metodo in METODOS_SEED_AGGREGATION:
#             rodar_etapa(f"{nome}__seed_aggregation__{metodo}", lambda cfg=cfg, dataset=dataset, metodo=metodo: etapa_seed_aggregation(cfg, dataset, metodo))

#         for metodo in METODOS:
#             rodar_etapa(f"{nome}__comparacao_arquiteturas__{metodo}", lambda cfg=cfg, dataset=dataset, metodo=metodo: etapa_comparacao_arquiteturas(cfg, dataset, metodo))

#         for metodo in METODOS_SANITY_CHECK:
#             rodar_etapa(f"{nome}__sanity_check__{metodo}", lambda cfg=cfg, dataset=dataset, metodo=metodo: etapa_sanity_check(cfg, dataset, metodo))

#     rodar_etapa("comparacao_datasets", lambda: etapa_comparacao_datasets(DATASETS, datasets_obj))

#     log("=" * 70)
#     log("ANÁLISE XAI COMPLETA FINALIZADA (ver analise_xai/_checkpoints/ pra saber o que rodou)")
#     log("=" * 70)


# if __name__ == "__main__":
#     main()


"""
run_analise_completa.py

Orquestrador único da Etapa "análise XAI completa".

Não reimplementa as análises existentes:
apenas chama, na ordem correta, as funções já existentes em xai/*.py.

IMPORTANTE:
    Este arquivo foi modificado para:

    1. Ser retomável através de checkpoints .OK.
    2. Preservar os checkpoints já existentes.
    3. Usar TODAS as amostras para GradCAM e IG.
    4. Usar poucas amostras para Occlusion, pois esse método é
       extremamente mais caro.
    5. Não modificar xai_metrics.py.
    6. Se uma etapa falhar, continuar para as próximas.
    7. Se o processo for interrompido, as etapas já concluídas
       permanecem protegidas pelos checkpoints.

PATCHES NECESSÁRIOS ANTES DE RODAR:

    1. xai/xai_metrics.py:
       normalise=True no quantus.MaxSensitivity

    2. xai/comparacao_arquiteturas.py:
       corrigir:
           else img_orig
       para:
           else img_rec

    3. xai/comparacao_datasets.py:
       "Queda_prob" -> "queda_prob"

    4. xai/comparacao_datasets.py:
       figzise -> figsize

ATENÇÃO:

    Não apague a pasta analise_xai/.

    Os checkpoints existentes serão reutilizados.

    Como o xai_metrics.py executa todas as seeds internamente,
    uma execução interrompida no meio de Occlusion não pode ser
    retomada exatamente daquela seed sem modificar xai_metrics.py.

    Portanto, a estratégia é:
        - preservar GradCAM já concluído;
        - preservar IG já concluído;
        - refazer Occlusion de forma muito menor;
        - continuar normalmente das próximas etapas.
"""

import os
import time
import traceback
import numpy as np
from datetime import datetime

import torch
import pandas as pd

from model.utils import (
    SEEDS,
    transform_originais,
    transform_recplot,
)
from model.dataset import EnsembleTestDataset

from xai.loader import (
    carregar_todos_os_branches,
    obter_camada_alvo,
    BRANCH_FILES,
)

from xai.attributions import (
    gerar_atribuicao,
    desnormalizar,
)

from xai.case_selection import (
    BRANCH_PARA_CENARIO_INDIVIDUAL,
    casos_por_confianca,
    casos_por_margem,
)

from xai.seed_aggregation import (
    mapa_agregado_para_amostra,
    figura_mapa_agregado,
    salvar_mapas_agregados,
    seed_mais_representativa,
)

from xai.threshold_analysis import (
    analisar_limiar_de_decisao,
    gerar_figura_exemplos,
    _heatmap_unico,
)

from xai.xai_metrics import xai_metrics_to_csv

from xai.comparacao_arquiteturas import (
    comparar_arquiteturas_em_lotes,
    cruzar_com_predicoes,
    resumo_concordancia_vs_ensemble,
)

from xai.comparacao_datasets import (
    importancia_por_banda_dataset,
    resumo_importancia_por_banda,
    comparar_importancia_entre_datasets,
    figura_comparacao_bandas,
)

from xai.sanity_check import (
    rodar_sanity_check_todos_branches,
    plotar_curva_sanidade,
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

RAIZ_SAIDA = "analise_xai2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# MÉTODOS
# ------------------------------------------------------------

METODOS = [
    "gradcam",
    "ig",
    "occlusion",
]

METODOS_SANITY_CHECK = [
    "gradcam",
    "ig",
    "occlusion",
]

# Seed aggregation continua apenas com GradCAM porque é muito caro.
METODOS_SEED_AGGREGATION = [
    "gradcam",
]


# ------------------------------------------------------------
# PARÂMETROS GERAIS
# ------------------------------------------------------------

MARGEM_MAX = 0.1

N_CONFIANTES = 8

# Para GradCAM e IG:
# None = todas as amostras disponíveis.
N_AMOSTRAS_XAI_METRICS = None

# ------------------------------------------------------------
# PARÂMETRO ESPECÍFICO PARA OCCLUSION
# ------------------------------------------------------------
#
# ESTE É O PRINCIPAL AJUSTE DESTA VERSÃO.
#
# Occlusion estava levando horas com 46 amostras.
# Agora serão utilizadas somente 6 amostras.
#
# Você pode aumentar depois para 8, 10 etc.
#
N_AMOSTRAS_XAI_METRICS_OCCLUSION = None  # None = todas as amostras disponíveis, mas é MUITO caro


# ------------------------------------------------------------
# OUTRAS ANÁLISES
# ------------------------------------------------------------

N_AMOSTRAS_COMPARACAO_ARQ = 20

N_AMOSTRAS_COMPARACAO_DATASETS = 6

N_AMOSTRAS_SEED_AGG_POR_CATEGORIA = 2


# ------------------------------------------------------------
# EXEMPLOS
# ------------------------------------------------------------

SEED_FIXA_EXEMPLOS = SEEDS[0]

SAMPLE_FIXA_EXEMPLO = 0


# ============================================================
# DATASETS
# ============================================================

DATASETS = [
    dict(
        nome="displasia",
        classes=["healthy", "severe"],
        num_classes=2,

        test_dir=(
            r"C:\Users\IFTM-ITB\Desktop\EnsembleFractal"
            r"\datasets\daniel_tentando\novo\RPnew\teste"
        ),

        models_dir="resultados_displasia/models",

        predicoes_csv="resultados_displasia/predicoes.csv",

        resultados_testes_csv=(
            "resultados_displasia/resultados_testes.csv"
        ),
    ),

    dict(
        nome="pulmao",
        classes=["aca_md", "nor", "scc_md"],
        num_classes=3,

        test_dir=(
            r"C:\Users\IFTM-ITB\Desktop\EnsembleFractal"
            r"\datasets\daniel_tentando\dataset_lung\teste"
        ),

        models_dir="resultados_pulmao/models",

        predicoes_csv="resultados_pulmao/predicoes.csv",

        resultados_testes_csv=(
            "resultados_pulmao/resultados_testes.csv"
        ),
    ),
]


# ============================================================
# CENÁRIOS
# ============================================================

CENARIO_ENSEMBLE_POR_REPRESENTACAO = {
    "orig": "MobileNet_Original + EffNet_Original",
    "recplot": "MobileNet_RecPlot + EffNet_RecPlot",
}


# ============================================================
# INFRAESTRUTURA
# ============================================================

def caminho_log():
    return os.path.join(
        RAIZ_SAIDA,
        "logs",
        "execucao.log",
    )


def log(msg):
    linha = (
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"{msg}"
    )

    print(linha)

    with open(
        caminho_log(),
        "a",
        encoding="utf-8",
    ) as f:
        f.write(linha + "\n")


def criar_estrutura_pastas():
    """
    Cria todas as subpastas antes da execução.

    IMPORTANTE:
        exist_ok=True significa que as pastas existentes
        não são apagadas.
    """

    subpastas_por_dataset = [
        "heatmaps_exemplo",
        "xai_metricas",
        "limiar",
        "seed_aggregation",
        "comparacao_arquiteturas",
        "sanity_check",
    ]

    os.makedirs(
        os.path.join(RAIZ_SAIDA, "logs"),
        exist_ok=True,
    )

    os.makedirs(
        os.path.join(RAIZ_SAIDA, "_checkpoints"),
        exist_ok=True,
    )

    os.makedirs(
        os.path.join(RAIZ_SAIDA, "comparacao_datasets"),
        exist_ok=True,
    )

    for cfg in DATASETS:

        nome = cfg["nome"]

        for sub in subpastas_por_dataset:

            for metodo in METODOS:

                os.makedirs(
                    os.path.join(
                        RAIZ_SAIDA,
                        nome,
                        sub,
                        metodo,
                    ),
                    exist_ok=True,
                )

    log(
        f"Estrutura de pastas verificada em "
        f"{RAIZ_SAIDA}/"
    )


# ============================================================
# CHECKPOINT
# ============================================================

def rodar_etapa(nome_etapa, func):
    """
    Executa uma etapa usando checkpoint.

    Se:
        analise_xai/_checkpoints/NOME.OK

    existir, a etapa não será executada novamente.

    Se ocorrer erro:
        - registra o erro;
        - NÃO cria o checkpoint;
        - continua para a próxima etapa.

    Assim, uma etapa que falhou poderá ser tentada novamente
    em uma execução posterior.
    """

    marcador = os.path.join(
        RAIZ_SAIDA,
        "_checkpoints",
        f"{nome_etapa}.OK",
    )

    if os.path.exists(marcador):

        log(
            f"[PULANDO] {nome_etapa} "
            f"(checkpoint já existe)"
        )

        return

    log(
        f"[INICIO] {nome_etapa}"
    )

    inicio = time.time()

    try:

        func()

        with open(
            marcador,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                f"concluída em "
                f"{datetime.now().isoformat()}\n"
            )

        log(
            f"[OK] {nome_etapa} — "
            f"{time.time() - inicio:.1f}s"
        )

    except Exception as e:

        log(
            f"[ERRO] {nome_etapa}: "
            f"{repr(e)}"
        )

        log(
            traceback.format_exc()
        )


# ============================================================
# DATASET
# ============================================================

def carregar_dataset(cfg):

    return EnsembleTestDataset(
        cfg["test_dir"],
        cfg["classes"],
        transform_original=transform_originais,
        transform_recplot=transform_recplot,
    )


# ============================================================
# HEATMAPS DE EXEMPLO
# ============================================================

def etapa_heatmaps_exemplo(
    cfg,
    dataset,    
    metodo,
):

    import matplotlib.pyplot as plt

    branches = carregar_todos_os_branches(
        SEED_FIXA_EXEMPLOS,
        cfg["num_classes"],
        models_dir=cfg["models_dir"],
        device=DEVICE,
    )

    img_orig, img_rec, label = dataset[
        SAMPLE_FIXA_EXEMPLO
    ]

    target_class = (
        int(label.item())
        if isinstance(label, torch.Tensor)
        else int(label)
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    for ax, (
        nome_branch,
        modelo,
    ) in zip(
        axes,
        branches.items(),
    ):

        if modelo is None:

            ax.axis("off")
            continue

        tipo_dataset = BRANCH_FILES[
            nome_branch
        ][1]

        input_tensor = (
            img_orig
            if tipo_dataset == "originais"
            else img_rec
        )

        mapa = _heatmap_unico(
            modelo,
            input_tensor,
            target_class,
            metodo=metodo,
        )

        fundo = desnormalizar(
            input_tensor,
            tipo=(
                "orig"
                if tipo_dataset == "originais"
                else "rec"
            ),
        )

        ax.imshow(fundo)
        if metodo == "ig":

            limiar = np.percentile(mapa, 95)

            
            alpha = np.where(
                mapa >= limiar,
                0.8,
                0.0
            )

            ax.imshow(
                mapa,
                cmap="jet",
                alpha=alpha,
                vmin=limiar,
                vmax=mapa.max(),
            )

        else:
            ax.imshow(
                mapa,
                cmap="jet",
                alpha=0.5,
            )

        ax.set_title(
            nome_branch,
            fontsize=10,
        )

        ax.axis("off")

    fig.suptitle(
        f"{cfg['nome']} — seed "
        f"{SEED_FIXA_EXEMPLOS} — amostra "
        f"{SAMPLE_FIXA_EXEMPLO} — {metodo}"
    )

    plt.tight_layout()

    caminho = os.path.join(
        RAIZ_SAIDA,
        cfg["nome"],
        "heatmaps_exemplo",
        metodo,
        "exemplo_4_branches.png",
    )

    fig.savefig(
        caminho,
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# XAI METRICS
# ============================================================

def etapa_xai_metrics(
    cfg,
    dataset,
    metodo,
):
    """
    Executa xai_metrics.py.

    A diferença importante desta versão:

        GradCAM -> todas as amostras
        IG      -> todas as amostras
        Occlusion -> somente
                     N_AMOSTRAS_XAI_METRICS_OCCLUSION

    Não modifica xai_metrics.py.
    """

    out_dir = os.path.join(
        RAIZ_SAIDA,
        cfg["nome"],
        "xai_metricas",
        metodo,
    )

    # --------------------------------------------------------
    # Escolha da quantidade de amostras
    # --------------------------------------------------------

    if metodo == "occlusion":

        n_amostras = (
            N_AMOSTRAS_XAI_METRICS_OCCLUSION
        )

        log(
            f"[XAI METRICS] {cfg['nome']} / "
            f"{metodo}: usando somente "
            f"{n_amostras} amostras "
            f"(redução de custo)"
        )

    else:

        n_amostras = (
            N_AMOSTRAS_XAI_METRICS
        )

        log(
            f"[XAI METRICS] {cfg['nome']} / "
            f"{metodo}: usando todas as "
            f"amostras"
        )

    # --------------------------------------------------------
    # Chamada original
    # --------------------------------------------------------

    xai_metrics_to_csv(
        seeds=SEEDS,
        num_classes=cfg["num_classes"],
        models_dir=cfg["models_dir"],
        dataset=dataset,
        output_dir=out_dir,
        metodo=metodo,
        n_amostras=n_amostras,
        device=DEVICE,
    )


# ============================================================
# LIMIAR
# ============================================================

def etapa_limiar(
    cfg,
    dataset,
    metodo,
):

    out_dir = os.path.join(
        RAIZ_SAIDA,
        cfg["nome"],
        "limiar",
        metodo,
    )

    df_resultado, df_resumo = (
        analisar_limiar_de_decisao(
            cfg["predicoes_csv"],
            dataset,
            cfg["nome"],
            cfg["num_classes"],
            out_dir,
            seeds=SEEDS,
            margem_max=MARGEM_MAX,
            n_confiantes=N_CONFIANTES,
            models_dir=cfg["models_dir"],
            metodo=metodo,
            device=DEVICE,
        )
    )

    for nome_branch in BRANCH_FILES:

        try:

            fig = gerar_figura_exemplos(
                dataset,
                df_resultado,
                nome_branch,
                SEED_FIXA_EXEMPLOS,
                metodo=metodo,
                models_dir=cfg["models_dir"],
                num_classes=cfg["num_classes"],
                device=DEVICE,
            )

            fig.savefig(
                os.path.join(
                    out_dir,
                    f"exemplos_{nome_branch}.png",
                ),
                dpi=150,
            )

        except Exception:

            log(
                f"[AVISO] figura de exemplos "
                f"falhou para "
                f"{nome_branch}/{metodo}"
            )

            log(
                traceback.format_exc(
                    limit=1
                )
            )


# ============================================================
# SEED AGGREGATION
# ============================================================

def etapa_seed_aggregation(
    cfg,
    dataset,
    metodo,
):

    df_pred = pd.read_csv(
        cfg["predicoes_csv"]
    )

    out_dir = os.path.join(
        RAIZ_SAIDA,
        cfg["nome"],
        "seed_aggregation",
        metodo,
    )

    for (
        nome_branch,
        cenario,
    ) in BRANCH_PARA_CENARIO_INDIVIDUAL.items():

        samples_acerto = casos_por_confianca(
            df_pred,
            SEED_FIXA_EXEMPLOS,
            cenario,
            tipo="acerto_confiante",
            n=N_AMOSTRAS_SEED_AGG_POR_CATEGORIA,
        )

        samples_erro = casos_por_confianca(
            df_pred,
            SEED_FIXA_EXEMPLOS,
            cenario,
            tipo="erro_confiante",
            n=N_AMOSTRAS_SEED_AGG_POR_CATEGORIA,
        )

        df_fronteira = casos_por_margem(
            df_pred,
            SEED_FIXA_EXEMPLOS,
            cenario,
            margem_max=MARGEM_MAX,
        )

        samples_fronteira = (
            df_fronteira[
                "sample"
            ]
            .head(
                N_AMOSTRAS_SEED_AGG_POR_CATEGORIA
            )
            .tolist()
        )

        amostras_marcadas = (
            [
                (s, "acerto_confiante")
                for s in samples_acerto
            ]
            +
            [
                (s, "erro_confiante")
                for s in samples_erro
            ]
            +
            [
                (s, "fronteira")
                for s in samples_fronteira
            ]
        )

        for (
            sample,
            categoria,
        ) in amostras_marcadas:

            linha = df_pred[
                (df_pred["seed"]
                 == SEED_FIXA_EXEMPLOS)
                &
                (df_pred["cenario"]
                 == cenario)
                &
                (df_pred["sample"]
                 == sample)
            ]

            if linha.empty:
                continue

            target_class = int(
                linha.iloc[0]["y_pred"]
            )

            img_orig, img_rec, label = (
                dataset[sample]
            )

            media, desvio, seeds_usadas = (
                mapa_agregado_para_amostra(
                    nome_branch,
                    img_orig,
                    img_rec,
                    target_class,
                    cfg["num_classes"],
                    seeds=SEEDS,
                    models_dir=cfg["models_dir"],
                    metodo=metodo,
                    device=DEVICE,
                )
            )

            nome_base = (
                f"{nome_branch}_"
                f"{categoria}_"
                f"amostra{sample}"
            )

            salvar_mapas_agregados(
                media,
                desvio,
                seeds_usadas,
                out_dir,
                nome_base,
            )

            tipo_dataset = BRANCH_FILES[
                nome_branch
            ][1]

            fundo = desnormalizar(
                img_orig
                if tipo_dataset == "originais"
                else img_rec,
                tipo=(
                    "orig"
                    if tipo_dataset == "originais"
                    else "rec"
                ),
            )

            fig = figura_mapa_agregado(
                media,
                desvio,
                img_fundo=fundo,
                titulo=(
                    f"{nome_branch} — "
                    f"{categoria} — "
                    f"amostra {sample}"
                ),
            )

            fig.savefig(
                os.path.join(
                    out_dir,
                    f"{nome_base}.png",
                ),
                dpi=150,
            )

            import matplotlib.pyplot as plt

            plt.close(fig)


# ============================================================
# COMPARAÇÃO DE ARQUITETURAS
# ============================================================

def etapa_comparacao_arquiteturas(
    cfg,
    dataset,
    metodo,
):

    df_pred = pd.read_csv(
        cfg["predicoes_csv"]
    )

    out_dir = os.path.join(
        RAIZ_SAIDA,
        cfg["nome"],
        "comparacao_arquiteturas",
        metodo,
    )

    n_disponivel = min(
        N_AMOSTRAS_COMPARACAO_ARQ,
        len(dataset),
    )

    indices = list(
        range(n_disponivel)
    )

    for (
        representacao,
        cenario_ensemble,
    ) in CENARIO_ENSEMBLE_POR_REPRESENTACAO.items():

        partes = []

        for seed in SEEDS:

            branches = (
                carregar_todos_os_branches(
                    seed,
                    cfg["num_classes"],
                    models_dir=cfg["models_dir"],
                    device=DEVICE,
                )
            )

            df_comp = (
                comparar_arquiteturas_em_lotes(
                    branches,
                    dataset,
                    indices,
                    representacao,
                    metodo=metodo,
                )
            )

            df_comp["seed"] = seed

            df_cruzado = (
                cruzar_com_predicoes(
                    df_comp,
                    df_pred,
                    seed,
                    cenario_ensemble,
                )
            )

            partes.append(
                df_cruzado
            )

        df_total = pd.concat(
            partes,
            ignore_index=True,
        )

        df_total.to_csv(
            os.path.join(
                out_dir,
                f"comparacao_{representacao}.csv",
            ),
            index=False,
        )

        df_resumo = (
            resumo_concordancia_vs_ensemble(
                df_total
            )
        )

        df_resumo.to_csv(
            os.path.join(
                out_dir,
                f"resumo_{representacao}.csv",
            )
        )


# ============================================================
# SANITY CHECK
# ============================================================

def etapa_sanity_check(
    cfg,
    dataset,
    metodo,
):

    branches = (
        carregar_todos_os_branches(
            SEED_FIXA_EXEMPLOS,
            cfg["num_classes"],
            models_dir=cfg["models_dir"],
            device=DEVICE,
        )
    )

    img_orig, img_rec, label = (
        dataset[SAMPLE_FIXA_EXEMPLO]
    )

    target_class = (
        int(label.item())
        if isinstance(label, torch.Tensor)
        else int(label)
    )

    resultados = (
        rodar_sanity_check_todos_branches(
            branches,
            img_orig,
            img_rec,
            target_class,
            metodo=metodo,
        )
    )

    out_dir = os.path.join(
        RAIZ_SAIDA,
        cfg["nome"],
        "sanity_check",
        metodo,
    )

    linhas = []

    for (
        nome_branch,
        lista,
    ) in resultados.items():

        for r in lista:

            linhas.append(
                {
                    "branch": nome_branch,
                    **r,
                }
            )

    pd.DataFrame(linhas).to_csv(
        os.path.join(
            out_dir,
            "sanity_check.csv",
        ),
        index=False,
    )

    import matplotlib.pyplot as plt

    for metrica in (
        "spearman",
        "ssim",
    ):

        fig = plotar_curva_sanidade(
            resultados,
            metrica=metrica,
            titulo=(
                f"{cfg['nome']} — "
                f"sanity check ({metodo}) — "
                f"{metrica}"
            ),
        )

        fig.savefig(
            os.path.join(
                out_dir,
                f"curva_{metrica}.png",
            ),
            dpi=150,
        )

        plt.close(fig)


# ============================================================
# COMPARAÇÃO ENTRE DATASETS
# ============================================================

def etapa_comparacao_datasets(
    datasets_cfg,
    datasets_obj,
):

    out_dir = os.path.join(
        RAIZ_SAIDA,
        "comparacao_datasets",
    )

    resumos = {}

    for nome_branch in [
        "mobilenet_recplot",
        "effnet_recplot",
    ]:

        resumos[nome_branch] = {}

        for (
            cfg,
            dataset,
        ) in zip(
            datasets_cfg,
            datasets_obj,
        ):

            n_disponivel = min(
                N_AMOSTRAS_COMPARACAO_DATASETS,
                len(dataset),
            )

            df_importancia = (
                importancia_por_banda_dataset(
                    nome_branch,
                    dataset,
                    list(
                        range(n_disponivel)
                    ),
                    cfg["num_classes"],
                    seeds=SEEDS,
                    models_dir=cfg["models_dir"],
                    device=DEVICE,
                )
            )

            df_importancia.to_csv(
                os.path.join(
                    out_dir,
                    f"importancia_"
                    f"{nome_branch}_"
                    f"{cfg['nome']}.csv",
                ),
                index=False,
            )

            df_resumo = (
                resumo_importancia_por_banda(
                    df_importancia
                )
            )

            df_resumo.to_csv(
                os.path.join(
                    out_dir,
                    f"resumo_"
                    f"{nome_branch}_"
                    f"{cfg['nome']}.csv",
                )
            )

            resumos[
                nome_branch
            ][cfg["nome"]] = df_resumo

        nome_a = datasets_cfg[0]["nome"]
        nome_b = datasets_cfg[1]["nome"]

        tabela = (
            comparar_importancia_entre_datasets(
                resumos[nome_branch][nome_a],
                resumos[nome_branch][nome_b],
                nome_a=nome_a,
                nome_b=nome_b,
            )
        )

        tabela.to_csv(
            os.path.join(
                out_dir,
                f"comparacao_"
                f"{nome_branch}_"
                f"{nome_a}_vs_{nome_b}.csv",
            )
        )

        fig = figura_comparacao_bandas(
            tabela,
            nome_a,
            nome_b,
        )

        fig.savefig(
            os.path.join(
                out_dir,
                f"figura_"
                f"{nome_branch}_"
                f"{nome_a}_vs_{nome_b}.png",
            ),
            dpi=150,
        )

        import matplotlib.pyplot as plt

        plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # NÃO APAGA NADA.
    # Apenas garante que as pastas existem.
    # --------------------------------------------------------

    criar_estrutura_pastas()

    log("=" * 70)
    log("INICIANDO / RETOMANDO ANÁLISE XAI COMPLETA")
    log("=" * 70)

    log(
        f"Dispositivo utilizado: {DEVICE}"
    )

    # --------------------------------------------------------
    # Carrega datasets
    # --------------------------------------------------------

    datasets_obj = []

    for cfg in DATASETS:

        if (
            not cfg.get("test_dir")
            or not os.path.exists(
                cfg["test_dir"]
            )
        ):

            log(
                f"[ERRO FATAL] test_dir de "
                f"{cfg['nome']} não existe:"
            )

            log(
                f"    {cfg['test_dir']}"
            )

            return

        log(
            f"Carregando dataset "
            f"{cfg['nome']}..."
        )

        datasets_obj.append(
            carregar_dataset(cfg)
        )

    # --------------------------------------------------------
    # PROCESSAMENTO POR DATASET
    # --------------------------------------------------------

    for (
        cfg,
        dataset,
    ) in zip(
        DATASETS,
        datasets_obj,
    ):

        nome = cfg["nome"]

        log(
            f"--- Dataset: {nome} "
            f"({len(dataset)} amostras de teste) ---"
        )

        # ====================================================
        # HEATMAPS
        # ====================================================

        for metodo in METODOS:

            rodar_etapa(
                f"{nome}__heatmaps_exemplo__{metodo}",
                lambda
                cfg=cfg,
                dataset=dataset,
                metodo=metodo:
                    etapa_heatmaps_exemplo(
                        cfg,
                        dataset,
                        metodo,
                    ),
            )

        # ====================================================
        # XAI METRICS
        #
        # IMPORTANTÍSSIMO:
        #
        # Se os checkpoints GradCAM e IG já existem,
        # eles serão simplesmente pulados.
        #
        # Occlusion não possui checkpoint e será executado
        # novamente, mas com somente 6 amostras.
        # ====================================================

        for metodo in METODOS:

            rodar_etapa(
                f"{nome}__xai_metrics__{metodo}",
                lambda
                cfg=cfg,
                dataset=dataset,
                metodo=metodo:
                    etapa_xai_metrics(
                        cfg,
                        dataset,
                        metodo,
                    ),
            )

        # ====================================================
        # LIMIAR
        # ====================================================

        for metodo in METODOS:

            rodar_etapa(
                f"{nome}__limiar__{metodo}",
                lambda
                cfg=cfg,
                dataset=dataset,
                metodo=metodo:
                    etapa_limiar(
                        cfg,
                        dataset,
                        metodo,
                    ),
            )

        # ====================================================
        # SEED AGGREGATION
        # ====================================================

        for metodo in METODOS_SEED_AGGREGATION:

            rodar_etapa(
                f"{nome}__seed_aggregation__{metodo}",
                lambda
                cfg=cfg,
                dataset=dataset,
                metodo=metodo:
                    etapa_seed_aggregation(
                        cfg,
                        dataset,
                        metodo,
                    ),
            )

        # ====================================================
        # COMPARAÇÃO DE ARQUITETURAS
        # ====================================================

        for metodo in METODOS:

            rodar_etapa(
                f"{nome}__comparacao_arquiteturas__{metodo}",
                lambda
                cfg=cfg,
                dataset=dataset,
                metodo=metodo:
                    etapa_comparacao_arquiteturas(
                        cfg,
                        dataset,
                        metodo,
                    ),
            )

        # ====================================================
        # SANITY CHECK
        # ====================================================

        for metodo in METODOS_SANITY_CHECK:

            rodar_etapa(
                f"{nome}__sanity_check__{metodo}",
                lambda
                cfg=cfg,
                dataset=dataset,
                metodo=metodo:
                    etapa_sanity_check(
                        cfg,
                        dataset,
                        metodo,
                    ),
            )

    # ========================================================
    # COMPARAÇÃO ENTRE DATASETS
    # ========================================================

    rodar_etapa(
        "comparacao_datasets",
        lambda:
            etapa_comparacao_datasets(
                DATASETS,
                datasets_obj,
            ),
    )

    log("=" * 70)

    log(
        "ANÁLISE XAI COMPLETA FINALIZADA."
    )

    log(
        "Confira analise_xai/_checkpoints/ "
        "para saber quais etapas foram concluídas."
    )

    log("=" * 70)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()