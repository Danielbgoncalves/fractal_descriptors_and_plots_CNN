from model.explainable.gradcam import *
from model.dataset import EnsembleTestDataset
# from metrics import metrics_to_csv
# from save_outputs import salvar_saidas_todos
from model.explainable.gradcam import *
# from efficiency import tabela_eficiencia
from model.utils import *
from model.dataset import *
# from train import *
from model.model import *

import torch
from torch.utils.data import DataLoader

import sys, os


model_orig = carregar_modelo('mobilenet', num_classes=2, path_weights=r"C:\Users\IFTM-ITB\Desktop\EnsembleFractal\models\42\mobilenet_originais.pth")
model_rec = carregar_modelo('mobilenet', num_classes=2, path_weights=r"C:\Users\IFTM-ITB\Desktop\EnsembleFractal\models\42\mobilenet_F-RecPlot.pth")
backbone = 'mobilenet'

test_dataset = EnsembleTestDataset("../../../datasets/dataset_displasia/teste", ["healthy", "severe"], transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
classes = ["healthy", "severe"]

N_GRADCAM = 30
gradcam_dir = "outputs/displasia/gradcam"

gerar_gradcam_comparacao(
    model_orig   = model_orig,
    model_rec    = model_rec,
    backbone     = backbone,
    loader       = test_loader,
    class_names  = classes,
    split        = "test",
    n_por_classe = N_GRADCAM,
    base_dir     = gradcam_dir,
)