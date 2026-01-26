# -*- coding: utf-8 -*-
import json
from pathlib import Path
from datetime import datetime, timezone
import cv2
from ultralytics import YOLO

# =========================
# Config MVP
# =========================
CLASS_NAME = "instrument"
CONF_TH = 0.35
FRAME_STRIDE = 5  # processa 1 a cada N frames
RISK_PERSISTENCE_SEC = 8  # risco se persistir por >= X segundos


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def sec_from_frame(frame_idx: int, fps: float) -> float:
    return frame_idx / fps if fps > 0 else 0.0


def utc_run_id() -> str:
    # ex.: 2026-01-26_235959Z
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")


def build_report(events: dict) -> str:
    vid = events["video_id"]
    created = events["created_at"]
    det_count = events["summary"]["detections_count"]
    risk_count = events["summary"]["risk_signals_count"]

    # Severidade máxima (se houver)
    sev_order = {"low": 0, "medium": 1, "high": 2}
    max_sev = "low"
    for rs in events.get("risk_signals", []):
        s = rs.get("severity", "low")
        if sev_order.get(s, 0) > sev_order.get(max_sev, 0):
            max_sev = s

    lines = []
    lines.append(f"# Evidências — Análise de Vídeo (YOLOv8)")
    lines.append("")
    lines.append(f"- **Vídeo:** `{vid}`")
    lines.append(f"- **Gerado em (UTC):** {created}")
    lines.append(f"- **Modelo:** {events['model']['name']} | weights: `{events['model']['weights']}` | conf_th={events['model']['conf_th']}")
    lines.append(f"- **FPS:** {events['video_meta']['fps']:.2f} | stride: {events['video_meta']['frame_stride']} | frames: {events['video_meta']['total_frames']}")
    lines.append("")
    lines.append("## Sumário")
    lines.append(f"- Detecções: **{det_count}**")
    lines.append(f"- Sinais de risco: **{risk_count}**")
    lines.append(f"- Severidade máxima: **{max_sev}**")
    lines.append("")
    lines.append("## Sinais de risco (detecção precoce)")
    if not events.get("risk_signals"):
        lines.append("- Nenhum sinal de risco gerado pelas regras atuais.")
    else:
        for rs in events["risk_signals"]:
            lines.append(
                f"- **{rs['severity']}** | {rs['reason']} | "
                f"{rs['t_start']}s → {rs['t_end']}s"
            )
    lines.append("")
    lines.append("## Observações técnicas")
    lines.append("- Este relatório é gerado por um MVP com regras simples de persistência temporal.")
    lines.append("- A lógica pode ser evoluída para modelos temporais/etapas clínicas e thresholds calibrados por especialistas.")
    lines.append("")
    return "\n".join(lines)


def main(
    video_path: str,
    weights_path: str,
    out_dir: str = "docs/evidencias-video",
    upload_azure: bool = False,
):
    video_path = str(video_path)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    frames_dir = out_dir / "frames_annotated"
    ensure_dir(frames_dir)

    model = YOLO(weights_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir vídeo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    detections = []
    presence_timestamps = []

    frame_idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue

        t = sec_from_frame(frame_idx, fps)
        results = model.predict(frame, conf=CONF_TH, verbose=False)
        r0 = results[0]

        has_any = False
        if r0.boxes is not None and len(r0.boxes) > 0:
            for b in r0.boxes:
                conf = float(b.conf[0].item())
                xyxy = b.xyxy[0].tolist()
                detections.append({
                    "t": round(t, 3),
                    "cls": CLASS_NAME,
                    "conf": round(conf, 4),
                    "bbox_xyxy": [round(x, 2) for x in xyxy],
                })
                has_any = True

            if saved < 15:
                annotated = r0.plot()
                out_img = frames_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_img), annotated)
                saved += 1

        if has_any:
            presence_timestamps.append(t)

        frame_idx += 1

    cap.release()

    # =========================
    # Regra simples de "detecção precoce de riscos"
    # =========================
    risk_signals = []
    if presence_timestamps:
        presence_timestamps.sort()
        start = presence_timestamps[0]
        prev = presence_timestamps[0]

        def severity_by_duration(dur: float) -> str:
            if dur >= 20:
                return "high"
            if dur >= RISK_PERSISTENCE_SEC:
                return "medium"
            return "low"

        gap = (FRAME_STRIDE / fps) * 1.5

        for t in presence_timestamps[1:]:
            if (t - prev) <= gap:
                prev = t
                continue

            dur = prev - start
            sev = severity_by_duration(dur)
            if sev != "low":
                risk_signals.append({
                    "type": "early_risk",
                    "severity": sev,
                    "reason": f"Persistência de {CLASS_NAME} por ~{dur:.1f}s",
                    "t_start": round(start, 3),
                    "t_end": round(prev, 3),
                })

            start = t
            prev = t

        dur = prev - start
        sev = severity_by_duration(dur)
        if sev != "low":
            risk_signals.append({
                "type": "early_risk",
                "severity": sev,
                "reason": f"Persistência de {CLASS_NAME} por ~{dur:.1f}s",
                "t_start": round(start, 3),
                "t_end": round(prev, 3),
            })

    run_id = utc_run_id()

    events = {
        "run_id": run_id,
        "video_id": Path(video_path).name,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {"type": "local", "uri": video_path},
        "model": {"name": "yolov8", "weights": weights_path, "conf_th": CONF_TH},
        "video_meta": {"fps": float(fps), "total_frames": total_frames, "frame_stride": FRAME_STRIDE},
        "detections": detections,
        "risk_signals": risk_signals,
        "summary": {
            "detections_count": len(detections),
            "risk_signals_count": len(risk_signals),
        },
    }

    out_json = out_dir / "events.json"
    out_json.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")

    out_report = out_dir / "report.md"
    out_report.write_text(build_report(events), encoding="utf-8")

    print(f"OK: {out_json}")
    print(f"OK: {out_report}")

    if upload_azure:
        from src.azure.storage import upload_file

        video_name = Path(video_path).stem
        base_prefix = f"runs/{run_id}/video/{video_name}"

        upload_file(str(out_json), f"{base_prefix}/events.json")
        upload_file(str(out_report), f"{base_prefix}/report.md")

        for img in frames_dir.glob("*.jpg"):
            upload_file(str(img), f"{base_prefix}/frames_annotated/{img.name}")

        print(f"OK: uploaded to Azure prefix: {base_prefix}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Caminho do vídeo de entrada")
    ap.add_argument("--weights", required=True, help="Pesos YOLO (best.pt)")
    ap.add_argument("--out", default="docs/evidencias-video", help="Diretório de saída")
    ap.add_argument("--upload-azure", action="store_true", help="Envia evidências para o Azure Blob")
    args = ap.parse_args()
    main(args.video, args.weights, args.out, args.upload_azure)
