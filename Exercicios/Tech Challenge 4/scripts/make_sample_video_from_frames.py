from pathlib import Path
import cv2

def main(
    frames_dir="data/yolo_endovis/images/train",
    out_video="data/raw/sample.mp4",
    fps=10,
    max_frames=300,
):
    frames_dir = Path(frames_dir)
    out_video = Path(out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    # pega imagens .jpg ordenadas
    imgs = sorted(frames_dir.glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"Nenhuma imagem .jpg encontrada em: {frames_dir}")

    imgs = imgs[:max_frames]

    # lê 1ª para pegar tamanho
    first = cv2.imread(str(imgs[0]))
    if first is None:
        raise RuntimeError(f"Falha ao ler imagem: {imgs[0]}")
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Não foi possível criar o VideoWriter (verifique codec mp4v).")

    for p in imgs:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)

    writer.release()
    print(f"OK: {out_video.resolve()} ({len(imgs)} frames @ {fps} FPS)")

if __name__ == "__main__":
    main()
