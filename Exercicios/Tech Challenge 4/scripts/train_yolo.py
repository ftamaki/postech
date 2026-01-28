# -*- coding: utf-8 -*-
from pathlib import Path
import shutil
from ultralytics import YOLO


def main(
    data_yaml: str = "data/yolo_endovis/dataset.yaml",
    base_model: str = "yolov8n.pt",
    imgsz: int = 640,
    epochs: int = 20,
    batch: int = 16,
    project: str = ".",  # com runs_dir='runs' e task='detect', salva em runs/detect/<name>
    name: str = "yolo_instrument_endovis",
    export_to: str = "models/yolo/best.pt",
):
    model = YOLO(base_model)

    results = model.train(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        project=project,
        name=name,
        task="detect",
        exist_ok=True,
    )

    # Diretório do run (retornado pelo Ultralytics)
    save_dir = getattr(results, "save_dir", None)
    if not save_dir:
        # Fallback (não esperado, mas mantém robustez)
        # Com task='detect' e runs_dir='runs', isso tende a cair em runs/detect/<name>
        save_dir = Path("runs") / "detect" / name

    run_dir = Path(save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt não encontrado em: {best}")

    out = Path(export_to)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)

    print(f"OK: best.pt copiado para {out.resolve()}")
    print(f"Run dir: {run_dir.resolve()}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo_endovis/dataset.yaml")
    ap.add_argument("--base", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default=".")
    ap.add_argument("--name", default="yolo_instrument_endovis")
    ap.add_argument("--export-to", default="models/yolo/best.pt")
    args = ap.parse_args()

    main(
        data_yaml=args.data,
        base_model=args.base,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
        export_to=args.export_to,
    )
