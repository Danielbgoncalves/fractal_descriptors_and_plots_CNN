import numpy as np
import pandas as pd
import torch
import os
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix
)

from .utils import DEVICE


def mostrar_metricas(nome_cenario, y_true, y_pred):

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    acertos = np.sum(y_true_arr == y_pred_arr)
    acertos_form = f"{acertos}/{len(y_pred)}"

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    specificities = []

    for i in range(n_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]

        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(spec)

    specificity = np.mean(specificities)

    return {
        "acertos": acertos_form,
        "acc": acc,
        "f1_macro": f1,
        "recall_macro": recall,
        "specificity_macro": specificity
    }

def metrics_to_csv(seeds, results, test_loader, output_dir):
    all_results = []          # CSV de métricas
    all_predictions = []      # CSV de predições

    # Definir os cenários
    cenarios = [
        ("MobileNet_Original", "MobileNet_RecPlot"),
        ("MobileNet_Original", "EffNet_Original"),
        ("MobileNet_Original", "EffNet_RecPlot"),
        ("MobileNet_RecPlot",  "EffNet_Original"),
        ("MobileNet_RecPlot",  "EffNet_RecPlot"),
        ("EffNet_Original",    "EffNet_RecPlot"),
        ("MobileNet_Original", "MobileNet_Original"),
        ("MobileNet_RecPlot",  "MobileNet_RecPlot"),
        ("EffNet_Original",    "EffNet_Original"),
        ("EffNet_RecPlot",     "EffNet_RecPlot")
    ]

    for seed in seeds:

        # Mapear modelos
        models = {
            "MobileNet_Original": results[seed]["mobnet_orig"],
            "MobileNet_RecPlot":  results[seed]["mobnet_recplot"],
            "EffNet_Original":    results[seed]["effnet_orig"],
            "EffNet_RecPlot":     results[seed]["effnet_recplot"]
        }

        # Tipo de entrada de cada modelo
        model_input = {
            "MobileNet_Original": "orig",
            "MobileNet_RecPlot":  "rec",
            "EffNet_Original":    "orig",
            "EffNet_RecPlot":     "rec"
        }

        for model in models.values():
            model.eval()

        y_true = []

        # Predições de cada cenário
        preds_cenarios = {c: [] for c in cenarios}

        sample_idx = 0

        with torch.no_grad():

            for imgs_orig, imgs_rec, labels in test_loader:

                imgs_orig = imgs_orig.to(DEVICE)
                imgs_rec = imgs_rec.to(DEVICE)

                labels_np = labels.cpu().numpy()

                # Guardar labels verdadeiros
                y_true.extend(labels_np)

                # Forward de todos os modelos
                outputs = {}

                for name, model in models.items():
                    if model_input[name] == "orig":
                        outputs[name] = F.softmax(model(imgs_orig), dim=1)
                    else:
                        outputs[name] = F.softmax(model(imgs_rec), dim=1)

                # Ensemble de cada cenário
                for cenario in cenarios:

                    soma = outputs[cenario[0]] + outputs[cenario[1]]

                    # -----------------------------
                    # Normalização das probabilidades
                    # -----------------------------
                    probs = soma.cpu().numpy()
                    probs = probs / probs.sum(axis=1, keepdims=True)

                    y_pred = np.argmax(probs, axis=1)

                    preds_cenarios[cenario].extend(y_pred)

                    # Salvar predições individuais
                    for i in range(len(labels_np)):

                        row = {
                            "seed": seed,
                            "cenario": " + ".join(cenario),
                            "sample": sample_idx + i,
                            "y_true": int(labels_np[i]),
                            "y_pred": int(y_pred[i])
                        }

                        # Salva uma coluna para cada classe
                        for classe in range(probs.shape[1]):
                            row[f"prob_{classe}"] = float(probs[i, classe])

                        all_predictions.append(row)

                sample_idx += batch_len

        # Calcular métricas de cada cenário
        for cenario, y_pred in preds_cenarios.items():

            metrics = mostrar_metricas(
                nome_cenario=" + ".join(cenario),
                y_true=y_true,
                y_pred=y_pred
            )

            metrics["seed"] = seed
            metrics["cenario"] = " + ".join(cenario)

            all_results.append(metrics)

    # ==========================================================
    # CSV 1 - Métricas
    # ==========================================================

    df_results = pd.DataFrame(all_results)

    output_path = os.path.join(output_dir, "resultados_testes.csv")
    df_results.to_csv(output_path, index=False)

    # ==========================================================
    # CSV 2 - Predições completas
    # ==========================================================

    df_predictions = pd.DataFrame(all_predictions)

    output_path = os.path.join(output_dir, "predicoes.csv")
    df_predictions.to_csv(output_path, index=False)

    # ==========================================================
    # CSV 3 - Média das métricas
    # ==========================================================

    df = df_results.drop(columns=["acertos"])

    df_mean_agrupado = (
        df.groupby("cenario")
        .agg(
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            f1_mean=("f1_macro", "mean"),
            f1_std=("f1_macro", "std"),
            recall_mean=("recall_macro", "mean"),
            recall_std=("recall_macro", "std"),
            spec_mean=("specificity_macro", "mean"),
            spec_std=("specificity_macro", "std"),
        )
        .reset_index()
    )

    output_path = os.path.join(output_dir, "resultados_testes_mean.csv")
    df_mean_agrupado.to_csv(output_path, index=False)
    