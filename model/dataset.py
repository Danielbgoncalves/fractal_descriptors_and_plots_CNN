from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

def load_data_from_folders(dir_data, class_names, reshape_type):
    data_list = []
    for class_index, class_name in enumerate(class_names):
        dir_class = Path(dir_data) / class_name / reshape_type

        if dir_class.exists():
            images = list(dir_class.glob('*.*'))
            print(f"Imagens encontradas em {class_name}: {len(images)}")
            for img_path in images:
                data_list.append((str(img_path), class_index))
        else:
            print(f"AVISO: Diretório {dir_class} não encontrado!!!!")

    return data_list

class ImageDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class EnsembleTestDataset(Dataset):
    def __init__(self, root_dir, class_names, transform_original=None, transform_recplot=None):
        self.root = Path(root_dir)
        self.transform_original = transform_original
        self.transform_recplot = transform_recplot
        self.class_names = class_names
        self.data = []

        valid_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

        for label_idx, class_name in enumerate(class_names):
            path_orig = self.root / class_name / "originais"
            path_rec  = self.root / class_name / "RP_perc"

            if not path_orig.exists() or not path_rec.exists():
                print(f"Diretórios não encontrados para classe {class_name}")
                continue

            # cria um dicionário nome_base -> caminho
            rec_dict = {
                p.stem: p
                for p in path_rec.iterdir()
                if p.is_file() and p.suffix.lower() in valid_exts
            }

            orig_files = sorted(
                p for p in path_orig.iterdir()
                if p.is_file() and p.suffix.lower() in valid_exts
            )

            for orig_path in orig_files:
                key = orig_path.stem  # nome sem extensão

                if key in rec_dict:
                    self.data.append({
                        "path_orig": str(orig_path),
                        "path_rec": str(rec_dict[key]),
                        "label": label_idx
                    })
                else:
                    print(f"Aviso: Par não encontrado para {orig_path.name}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_orig = Image.open(item["path_orig"]).convert("RGB")
        img_rec  = Image.open(item["path_rec"]).convert("RGB")
        label = item["label"]

        if self.transform_original:
            img_orig = self.transform_original(img_orig)
        if self.transform_recplot:
            img_rec = self.transform_recplot(img_rec)

        return img_orig, img_rec, label
