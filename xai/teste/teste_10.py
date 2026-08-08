from xai.threshold_analysis import analisar_limiar_de_decisao, gerar_figura_exemplos
from model.dataset import EnsembleTestDataset
from model.utils import transform_originais, transform_recplot

displasia_test_dir = r"C:\\Users\\IFTM-ITB\\Desktop\\EnsembleFractal\\datasets\\daniel_tentando\\novo\\RPnew\\teste"
lung_test_dir = r"C:\\Users\\IFTM-ITB\\Desktop\\EnsembleFractal\\datasets\\daniel_tentando\\dataset_lung\\teste"

for nome_dataset, classes, test_dir, predicoes_csv, num_classes, output_dir, models_dir in [
    ("displasia", ["healthy", "severe"], displasia_test_dir,
     r"resultados_displasia\predicoes.csv", 2, r"resultados_displasia\Gradcam", r"resultados_displasia\models"),

    ("pulmao", ["aca_md", "nor", "scc_md"], lung_test_dir,
     r"resultados_pulmao\predicoes.csv", 3, r"resultados_pulmao\Gradcam", r"resultados_pulmao\models"),
]:
    test_dataset = EnsembleTestDataset(test_dir, classes, transform_originais, transform_recplot)

    df_resultado, df_resumo = analisar_limiar_de_decisao(
        predicoes_csv, test_dataset, nome_dataset, num_classes, output_dir,
        margem_max=0.1, n_confiantes=15,models_dir=models_dir
    )
    print(df_resumo)

    fig = gerar_figura_exemplos(test_dataset, df_resultado, "mobilenet_recplot", seed=7,models_dir=models_dir, num_classes=num_classes)
    fig.savefig(f"{output_dir}/xai_figuras/limiar_mobilenet_recplot_seed7.png", dpi=150)