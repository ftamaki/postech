import os
from pathlib import Path
import random
import yaml
import numpy as np
import cv2
from PIL import Image
from datasets import load_dataset

# =========================
# Config
# =========================
HF_DATASET = "tyluan/Endovis2017"  # :contentReference[oaicite:5]{index=5}
OUT_DIR = Path("data/yolo_endovis")
IMG_SIZE = None  # ex.: (640, 640) se quiser redimensionar; senão mantém original
TRAIN_SPLIT = 0.9
SEED = 42

# Você pode começar com classe única: "instrument"
CLASSES = ["instrument"]


def mask_to_bbox(mask: np.ndarray):
    """
    Recebe máscara 2D (0=background; >0 instrumento) e retorna bbox (x_min, y_min, x_max, y_max).
    Estratégia simples: pega o maior contorno não-zero.
    """
    # binariza
    bin_mask = (mask > 0).astype(np.uint8) * 255
    if bin_mask.sum() == 0:
        return None

    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # maior contorno
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return (x, y, x + w, y + h)


def xyxy_to_yolo(x1, y1, x2, y2, w_img, h_img):
    """
    Converte bbox para YOLO format:
    class x_center y_center width height (normalizados 0-1)
    """
    bw = x2 - x1
    bh = y2 - y1
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    return (
        xc / w_img,
        yc / h_img,
        bw / w_img,
        bh / h_img,
    )


def ensure_dirs():
    for split in ["train", "val"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def save_yaml():
    data = {
        "path": str(OUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASSES)},
    }
    with open(OUT_DIR / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    ensure_dirs()
    save_yaml()

    # Carrega splits disponíveis no dataset (geralmente train/test; vamos refazer train/val localmente)
    ds = load_dataset(HF_DATASET)
    # Alguns datasets têm só "train". Se existir "test", juntamos tudo e redividimos.
    all_rows = []
    for split_name in ds.keys():
        all_rows.extend(ds[split_name])

    random.shuffle(all_rows)
    n_train = int(len(all_rows) * TRAIN_SPLIT)
    train_rows = all_rows[:n_train]
    val_rows = all_rows[n_train:]

    def process_rows(rows, split):
        for i, row in enumerate(rows):
            img: Image.Image = row["image"]
            lbl: Image.Image = row["label"]

            img_np = np.array(img)
            mask_np = np.array(lbl)

            # garante 2D na máscara
            if mask_np.ndim == 3:
                # se vier RGB, converte para cinza
                mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)

            if IMG_SIZE is not None:
                img_np = cv2.resize(img_np, IMG_SIZE, interpolation=cv2.INTER_AREA)
                mask_np = cv2.resize(mask_np, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

            h, w = img_np.shape[:2]
            bbox = mask_to_bbox(mask_np)
            if bbox is None:
                # Se não houver instrumento, você pode:
                # - pular, ou
                # - criar label vazio (YOLO aceita arquivo vazio)
                # Aqui: cria label vazio e salva imagem, útil para "negativos"
                x1 = y1 = x2 = y2 = None
                yolo_lines = []
            else:
                x1, y1, x2, y2 = bbox
                xc, yc, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
                # classe 0 = instrument
                yolo_lines = [f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"]

            # nomes
            stem = f"endovis_{split}_{i:06d}"
            img_path = OUT_DIR / "images" / split / f"{stem}.jpg"
            lbl_path = OUT_DIR / "labels" / split / f"{stem}.txt"

            # salva imagem
            Image.fromarray(img_np).save(img_path, quality=95)

            # salva label
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

    process_rows(train_rows, "train")
    process_rows(val_rows, "val")

    print("OK.")
    print(f"Dataset YOLO pronto em: {OUT_DIR.resolve()}")
    print(f"YAML: {OUT_DIR.resolve() / 'dataset.yaml'}")
    print("Treino (exemplo): yolo detect train data=data/yolo_endovis/dataset.yaml model=yolov8n.pt imgsz=640")


if __name__ == "__main__":
    main()
