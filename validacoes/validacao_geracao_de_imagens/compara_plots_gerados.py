import os
import cv2
import csv
from skimage.metrics import structural_similarity as ssim

def compare_images(base_python, base_matlab, output_csv="resultado_comparacao.csv", threshold=0.95):
    results = []

    subfolders = sorted(set(os.listdir(base_python)) & set(os.listdir(base_matlab)))

    if not subfolders:
        print("Nenhuma subpasta em comum encontrada.")
        return

    for sub in subfolders:
        path_py = os.path.join(base_python, sub)
        path_mat = os.path.join(base_matlab, sub)

        if not os.path.isdir(path_py) or not os.path.isdir(path_mat):
            continue

        files_py = set(os.listdir(path_py))
        files_mat = set(os.listdir(path_mat))
        common_files = sorted(files_py & files_mat)

        print(f"\n Subpasta: {sub}")

        for fname in common_files:
            f_py = os.path.join(path_py, fname)
            f_mat = os.path.join(path_mat, fname)

            img1 = cv2.imread(f_py)
            img2 = cv2.imread(f_mat)

            if img1 is None or img2 is None:
                print(f"    Erro ao ler: {fname}")
                continue

            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            if img1.shape != img2.shape:
                print(f"    Tamanho diferente: {fname} → ignorado")
                status = "dimensoes_diferentes"
                score = None
            else:
                # normalização
                img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
                img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)

                score = ssim(img1, img2, channel_axis=2)

                status = "equivalente" if score >= threshold else "diferente"

                print(f"  {fname} → SSIM: {score:.4f} → {status}")

            results.append([
                sub,
                fname,
                score if score is not None else "",
                status
            ])

    # salvar CSV
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subpasta", "arquivo", "ssim", "status"])
        writer.writerows(results)

    print(f"\n📄 Resultado salvo em: {output_csv}")


# ==================== USO =====================

base_python = r"codigo-fractal-python\feature_to_image\saida3"
base_matlab = r"codigos-fractal-matlab\saida3"

compare_images(base_python, base_matlab)