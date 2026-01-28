from pathlib import Path
import random
import cv2

def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return int(x1), int(y1), int(x2), int(y2)

def main(root="data/yolo_endovis", split="train", n=20, out="docs/dataset-check", seed=42):
    random.seed(seed)

    root = Path(root)
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    out_dir = Path(out) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not imgs:
        raise FileNotFoundError(f"Nenhuma imagem em {img_dir}")

    sample = random.sample(imgs, k=min(n, len(imgs)))

    for img_path in sample:
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        if lbl_path.exists():
            lines = [ln.strip() for ln in lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        else:
            lines = []

        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = parts
            xc = float(xc); yc = float(yc); bw = float(bw); bh = float(bh)

            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)
            # clamp
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"cls={cls}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        out_path = out_dir / f"{stem}_boxed.jpg"
        cv2.imwrite(str(out_path), img)

    print(f"OK: previews geradas em {out_dir.resolve()}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/yolo_endovis")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="docs/dataset-check")
    args = ap.parse_args()
    main(args.root, args.split, args.n, args.out)
