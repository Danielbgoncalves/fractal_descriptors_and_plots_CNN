import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import random
from pathlib import Path
import os

import torch.nn.functional as F

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
K_FOLDS = 5
SEEDS = [7, 12, 42, 65, 87, 93, 107, 121]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Utilizando dispositivo", DEVICE)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])