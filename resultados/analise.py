import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# Configurações
# ===========================

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11

# ===========================
# Função principal
# ===========================

def analisar_csv(csv_path, nome_dataset):

    df = pd.read_csv(csv_path)

    pasta_saida = f"plots_{nome_dataset.replace(' ','_')}"
    os.makedirs(pasta_saida, exist_ok=True)

    metricas = [
        "acc",
        "f1_macro",
        "recall_macro",
        "specificity_macro"
    ]

    # ----------------------------------------------------
    # Informações gerais
    # ----------------------------------------------------

    print("="*60)
    print(nome_dataset)
    print("="*60)

    print("\nNúmero de cenários:", df["cenario"].nunique())
    print("Número de seeds:", df["seed"].nunique())

    print("\nResumo:")
    print(df.describe())

    # ----------------------------------------------------
    # Ranking das médias
    # ----------------------------------------------------

    ranking = (
        df.groupby("cenario")[metricas]
        .agg(["mean","std"])
    )

    print("\nRanking:")
    print(ranking)

    ranking.to_csv(
        os.path.join(pasta_saida, "ranking_medias.csv")
    )

    # ----------------------------------------------------
    # Boxplots
    # ----------------------------------------------------

    for metrica in metricas:

        plt.figure(figsize=(14,6))

        sns.boxplot(
            data=df,
            x="cenario",
            y=metrica,
            color="lightgray"
        )

        sns.stripplot(
            data=df,
            x="cenario",
            y=metrica,
            color="red",
            size=5,
            jitter=True
        )

        plt.xticks(rotation=60, ha="right")
        plt.xlabel("")
        plt.ylabel(metrica)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                pasta_saida,
                f"boxplot_{metrica}.png"
            )
        )

        plt.close()

    # ----------------------------------------------------
    # Barplot das médias
    # ----------------------------------------------------

    for metrica in metricas:

        medias = (
            df.groupby("cenario")[metrica]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        plt.figure(figsize=(10,6))

        sns.barplot(
            data=medias,
            x=metrica,
            y="cenario"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                pasta_saida,
                f"ranking_{metrica}.png"
            )
        )

        plt.close()

    # ----------------------------------------------------
    # Heatmap das médias
    # ----------------------------------------------------

    medias = (
        df.groupby("cenario")[metricas]
        .mean()
    )

    plt.figure(figsize=(8,10))

    sns.heatmap(
        medias,
        annot=True,
        cmap="viridis",
        fmt=".3f"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            pasta_saida,
            "heatmap_medias.png"
        )
    )

    plt.close()

    # ----------------------------------------------------
    # Correlação
    # ----------------------------------------------------

    plt.figure(figsize=(6,5))

    sns.heatmap(
        df[metricas].corr(),
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            pasta_saida,
            "correlacao_metricas.png"
        )
    )

    plt.close()

    # ----------------------------------------------------
    # Histogramas
    # ----------------------------------------------------

    fig, axes = plt.subplots(2,2, figsize=(10,8))

    for ax, metrica in zip(axes.flat, metricas):

        sns.histplot(
            df[metrica],
            kde=True,
            ax=ax
        )

        ax.set_title(metrica)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            pasta_saida,
            "histogramas.png"
        )
    )

    plt.close()

    # ----------------------------------------------------
    # Violin plots
    # ----------------------------------------------------

    for metrica in metricas:

        plt.figure(figsize=(14,6))

        sns.violinplot(
            data=df,
            x="cenario",
            y=metrica
        )

        plt.xticks(rotation=60, ha="right")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                pasta_saida,
                f"violin_{metrica}.png"
            )
        )

        plt.close()

    # ----------------------------------------------------
    # Scatter Accuracy x F1
    # ----------------------------------------------------

    plt.figure(figsize=(7,6))

    sns.scatterplot(
        data=df,
        x="acc",
        y="f1_macro",
        hue="cenario",
        s=70
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            pasta_saida,
            "acc_vs_f1.png"
        )
    )

    plt.close()

    print("\nTodos os gráficos foram salvos em:", pasta_saida)


# ====================================================
# CHAMADAS
# ====================================================

analisar_csv(
    "resultados/teste_sempretreino/CSVs/resultados_lung.csv",
    "Lung_3_classes"
)

# analisar_csv(
#     "resultados/CSVs/resultados_testes_lung.csv",
#     "Lung_3_classes"
# )