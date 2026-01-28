from pathlib import Path
import shutil

FILES = [
    "results.png",
    "args.yaml",
    "results.csv",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
]

GLOBS = [
    "val_batch*_pred.jpg",
    "val_batch*_labels.jpg",
    "train_batch*.jpg",
]

def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False

def main(run_dir: str, out_dir: str):
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0

    for f in FILES:
        copied += int(copy_if_exists(run_dir / f, out_dir / f))

    for pattern in GLOBS:
        for p in run_dir.glob(pattern):
            dest = out_dir / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            copied += 1

    print(f"OK: {copied} arquivos copiados para {out_dir.resolve()}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default="docs/evidencias-video/treino/yolo_instrument_endovis")
    args = ap.parse_args()
    main(args.run_dir, args.out_dir)
