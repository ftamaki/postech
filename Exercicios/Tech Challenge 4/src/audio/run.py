# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from .utils import ensure_dir, mp3_to_wav_16k_mono
from .analyzers import build_events, build_report


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")


def transcribe_with_whisper(wav_path: Path, model_name: str = "small", language: str = "pt"):
    """
    Transcreve usando openai-whisper (local). Retorna (transcript, segments[])
    segments: [{"start": float, "end": float, "text": str}, ...]
    """
    import whisper  # lazy import

    model = whisper.load_model(model_name)
    # fp16 True em GPU, False em CPU (whisper decide, mas deixamos explícito)
    result = model.transcribe(
        str(wav_path),
        language=language,
        fp16=True,
        verbose=False,
    )

    transcript = (result.get("text") or "").strip()
    segments_out = []
    for s in result.get("segments", []) or []:
        segments_out.append({
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": (s.get("text") or "").strip(),
        })
    return transcript, segments_out


def main(
    audio_mp3: str,
    out_dir: str = "docs/evidencias-audio",
    model_name: str = "small",
    language: str = "pt",
    upload_azure: bool = False,
):
    audio_mp3 = Path(audio_mp3)
    if not audio_mp3.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {audio_mp3}")

    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    run_id = utc_run_id()

    # Converter MP3 -> WAV 16k mono
    wav_dir = out_dir / "tmp"
    ensure_dir(wav_dir)
    wav_path = wav_dir / f"{audio_mp3.stem}_16k.wav"
    mp3_to_wav_16k_mono(audio_mp3, wav_path)

    # Transcrever
    transcript, segments = transcribe_with_whisper(wav_path, model_name=model_name, language=language)

    # Construir eventos + report
    events = build_events(
        audio_path=str(audio_mp3.name),
        model_name=model_name,
        language=language,
        transcript=transcript,
        segments=segments,
        run_id=run_id,
    )

    out_json = out_dir / "events.json"
    out_json.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")

    out_report = out_dir / "report.md"
    out_report.write_text(build_report(events), encoding="utf-8")

    print(f"OK: {out_json}")
    print(f"OK: {out_report}")

    # Upload Azure (opcional)
    if upload_azure:
        from src.azure.storage import upload_file

        audio_name = audio_mp3.stem
        base_prefix = f"runs/{run_id}/audio/{audio_name}"
        upload_file(str(out_json), f"{base_prefix}/events.json")
        upload_file(str(out_report), f"{base_prefix}/report.md")

        # Subir também o mp3 original (útil para auditoria)
        upload_file(str(audio_mp3), f"{base_prefix}/input/{audio_mp3.name}")

        print(f"OK: uploaded to Azure prefix: {base_prefix}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Caminho do MP3 de entrada")
    ap.add_argument("--out", default="docs/evidencias-audio", help="Diretório de saída")
    ap.add_argument("--model", default="small", help="Whisper model: tiny/base/small/medium/large")
    ap.add_argument("--lang", default="pt", help="Idioma (ex.: pt, en)")
    ap.add_argument("--upload-azure", action="store_true", help="Envia evidências para Azure Blob")
    args = ap.parse_args()

    main(
        audio_mp3=args.audio,
        out_dir=args.out,
        model_name=args.model,
        language=args.lang,
        upload_azure=args.upload_azure,
    )
