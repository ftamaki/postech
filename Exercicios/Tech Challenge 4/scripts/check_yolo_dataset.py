from pathlib import Path
import re

def read_lines(p: Path):
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

def main(root="data/yolo_endovis", split="train", n_classes=1):
    root = Path(root)
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split

    imgs = {p.stem: p for p in img_dir.glob("*.jpg")}
    imgs |= {p.stem: p for p in img_dir.glob("*.png")}
    lbls = {p.stem: p for p in lbl_dir.glob("*.txt")}

    missing_labels = sorted(set(imgs.keys()) - set(lbls.keys()))
    missing_images = sorted(set(lbls.keys()) - set(imgs.keys()))

    print(f"[{split}] imagens: {len(imgs)} | labels: {len(lbls)}")
    print(f"Labels faltando para imagens: {len(missing_labels)}")
    print(f"Imagens faltando para labels: {len(missing_images)}")

    bad_files = 0
    bad_lines = 0

    float_re = re.compile(r"^-?\d+(\.\d+)?$")

    for stem, lp in lbls.items():
        lines = read_lines(lp)
        for i, ln in enumerate(lines, 1):
            parts = ln.split()
            if len(parts) != 5:
                print(f"[ERRO] {lp} linha {i}: esperado 5 colunas, veio {len(parts)} -> {ln}")
                bad_lines += 1
                continue

            cls_s, xc_s, yc_s, w_s, h_s = parts
            if not cls_s.isdigit():
                print(f"[ERRO] {lp} linha {i}: class_id inválido -> {cls_s}")
                bad_lines += 1
                continue

            cls = int(cls_s)
            if not (0 <= cls < n_classes):
                print(f"[ERRO] {lp} linha {i}: class_id fora do range -> {cls}")
                bad_lines += 1
                continue

            for name, val_s in [("xc", xc_s), ("yc", yc_s), ("w", w_s), ("h", h_s)]:
                if not float_re.match(val_s):
                    print(f"[ERRO] {lp} linha {i}: {name} não é número -> {val_s}")
                    bad_lines += 1
                    break
                val = float(val_s)
                if not (0.0 <= val <= 1.0):
                    print(f"[ERRO] {lp} linha {i}: {name} fora de 0..1 -> {val}")
                    bad_lines += 1
                    break

        # arquivo vazio é permitido (imagem negativa), então não conta como erro

    if missing_labels[:10]:
        print("Exemplos (labels faltando):", missing_labels[:10])
    if missing_images[:10]:
        print("Exemplos (imagens faltando):", missing_images[:10])

    if bad_lines == 0:
        print("OK: formato de labels parece consistente.")
    else:
        print(f"ATENÇÃO: {bad_lines} linhas com problema.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/yolo_endovis")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--n-classes", type=int, default=1)
    args = ap.parse_args()
    main(args.root, args.split, args.n_classes)
