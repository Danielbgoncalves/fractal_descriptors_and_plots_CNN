from .model import criar_modelo

EPOCHS = 30
SEED = 42

import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.models as models
import os
import pandas as pd

from .dataset import load_data_from_folders, ImageDataset
from sklearn.model_selection import StratifiedKFold
from .utils import *
from torch.utils.data import DataLoader


def train_one_fold(model, train_loader, val_loader, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    # Histórico por época
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    metrics = {
        "best_epoch": -1,
        "best_val_loss": float("inf"),
        "best_val_accuracy": -1,
        "best_train_loss": float("inf"),
        "best_train_accuracy": -1
    }

    best_model_state = None

    for epoch in range(epochs):
        # ================= TREINO =================
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)

        # ================= VALIDAÇÃO =================
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels).item()
                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        # ================= MELHOR MODELO =================
        if val_loss < metrics["best_val_loss"]:
            metrics["best_val_loss"] = val_loss
            best_model_state = model.state_dict()
            metrics["best_epoch"] = epoch+1
            metrics["best_val_accuracy"] = val_acc
            metrics["best_train_loss"] = train_loss
            metrics["best_train_accuracy"] = train_acc

        print(
            f"Época {epoch+1}/{epochs} | "
            f"TrainLoss {train_loss:.4f} | TrainAcc {train_acc:.4f} | "
            f"ValLoss {val_loss:.4f} | ValAcc {val_acc:.4f}"
        )

    return model, best_model_state, history, metrics

def run_kfold(dataset_path, dataset_type, class_names, backbone="mobilenet", seed=SEED, output_dir="resultados"):

    # os.makedirs(f"models/{seed}", exist_ok=True)

    data_list = load_data_from_folders(dataset_path, class_names, dataset_type)
    print(f"Total de imagens: {len(data_list)}")

    paths = [x[0] for x in data_list]
    labels = [x[1] for x in data_list]

    skf = StratifiedKFold(
        n_splits=K_FOLDS,
        shuffle=True,
        random_state=seed
    )

    fold_results = []
    all_history = []

    best_val_loss_global = float("inf")
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
        print("\n============================")
        print(f"FOLD {fold+1}/{K_FOLDS}")
        print("============================")

        train_data = [(paths[i], labels[i]) for i in train_idx]
        val_data   = [(paths[i], labels[i]) for i in val_idx]

        train_dataset = ImageDataset(train_data, transform=transform)
        val_dataset   = ImageDataset(val_data, transform=transform)

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

        # modelo
        model = criar_modelo(
            backbone=backbone,
            num_classes=len(class_names),
            pretrained='DEFAULT'
        ).to(DEVICE)

        model, best_model_fold_state, history, metrics = train_one_fold(
            model, train_loader, val_loader
        )

        # ---------- salvar melhor modelo global ----------
        if metrics["best_val_loss"] < best_val_loss_global:
            best_val_loss_global = metrics["best_val_loss"]
            best_model_state = best_model_fold_state

            dir = os.path.join(output_dir, "models", str(seed))
            os.makedirs(dir, exist_ok=True)
            save_path = os.path.join(dir, f"{backbone}_{dataset_type}.pth")
            torch.save(best_model_state, save_path)
            print(f"   -> Novo melhor modelo salvo! ValLoss: {best_val_loss_global:.4f}")

        # ---------- histórico e resultados ----------
        fold_results.append({
            "fold": fold + 1,
            **metrics
        })

        # adicionar histórico por época
        all_history.append({
            "fold": fold + 1,
            **history
        })

        del model
        torch.cuda.empty_cache()

    # ================= SALVAR RESULTADOS =================
    base_path = os.path.join(output_dir, "results_kfold", str(seed), dataset_type, backbone)
    os.makedirs(base_path, exist_ok=True)

    # métricas finais por fold
    df_metrics = pd.DataFrame(fold_results)
    df_metrics.to_csv(f"{base_path}/metrics.csv", index=False)

    # histórico completo
    df_history = pd.DataFrame(all_history)
    df_history.to_csv(f"{base_path}/history.csv", index=False)

    print("\n===== RESULTADOS FINAIS DO K-FOLD =====")
    print(df_metrics)
    print("\nMédias:")
    print(df_metrics.mean(numeric_only=True))

    # ================= RECRIAR MELHOR MODELO =================
    best_model = criar_modelo(
        backbone=backbone,
        num_classes=len(class_names),
        pretrained=None
    ).to(DEVICE)
    best_model.load_state_dict(best_model_state)

    return best_model


def train_seeds(seeds, dataset_path, classes, output_dir):
    for seed in seeds:
        print(f"\n===== Treinando com seed {seed} =====")
        set_seed(seed)

        results = {}
        results[seed] = {}

        # MobileNet - RP_perc
        results[seed]['mobnet_recplot'] = run_kfold(
            dataset_path, "RP_perc", classes, backbone="mobilenet", seed=seed, output_dir=output_dir
        )
        torch.cuda.empty_cache()

        # EfficientNet-B0 - RP_perc
        results[seed]['effnet_recplot'] = run_kfold(
            dataset_path, "RP_perc", classes, backbone="efficientnet_b0", seed=seed, output_dir=output_dir
        )
        torch.cuda.empty_cache()

        # MobileNet - originais
        results[seed]['mobnet_orig'] = run_kfold(
            dataset_path, "originais", classes, backbone="mobilenet", seed=seed, output_dir=output_dir
        )
        torch.cuda.empty_cache()

        # EfficientNet-B0 - originais
        results[seed]['effnet_orig'] = run_kfold(
            dataset_path, "originais", classes, backbone="efficientnet_b0", seed=seed, output_dir=output_dir
        )
        torch.cuda.empty_cache()

    return results