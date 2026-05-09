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
    all_results = []  # Aqui vamos armazenar todas as métricas para CSV

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
        ("EffNet_RecPlot",    "EffNet_RecPlot")
    ]

    for seed in seeds:

        # Mapear modelos
        models = {
            "MobileNet_Original": results[seed]['mobnet_orig'],
            "MobileNet_RecPlot":  results[seed]['mobnet_recplot'],
            "EffNet_Original":    results[seed]['effnet_orig'],
            "EffNet_RecPlot":     results[seed]['effnet_recplot']
        }

        # Mapear tipo de entrada
        model_input = {
            "MobileNet_Original": "orig",
            "MobileNet_RecPlot":  "rec",
            "EffNet_Original":    "orig",
            "EffNet_RecPlot":     "rec"
        }

        for model in models.values():
            model.eval()

        y_true = []

        # Inicializar dicionário para guardar predições
        preds_cenarios = {c: [] for c in cenarios}

        with torch.no_grad():
            for imgs_orig, imgs_rec, labels in test_loader:
                imgs_orig = imgs_orig.to(DEVICE)
                imgs_rec  = imgs_rec.to(DEVICE)

                # Forward de todos os modelos
                outputs = {}
                for name, model in models.items():
                    if model_input[name] == "orig":
                        outputs[name] = F.softmax(model(imgs_orig), dim=1)
                    else:
                        outputs[name] = F.softmax(model(imgs_rec), dim=1)

                # Guardar labels reais
                y_true.extend(labels.cpu().numpy())

                # Calcular predições para cada cenário
                for cenario in cenarios:
                    soma = outputs[cenario[0]] + outputs[cenario[1]]
                    y_pred = torch.argmax(soma, dim=1).cpu().numpy()
                    preds_cenarios[cenario].extend(y_pred)

        # Calcular métricas e armazenar para CSV
        for cenario, y_pred in preds_cenarios.items():
            # calcular métricas usando sua função mostrar_metricas
            metrics = mostrar_metricas(nome_cenario=" + ".join(cenario), y_true=y_true, y_pred=y_pred)

            metrics["seed"] = seed
            metrics["cenario"] = " + ".join(cenario)

            all_results.append(metrics)

    # Criar DataFrame e salvar CSV
    df_results = pd.DataFrame(all_results)
    output_path = os.path.join(output_dir, "resultados_testes.csv")
    df_results.to_csv(output_path, index=False)