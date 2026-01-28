# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import subprocess


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def mp3_to_wav_16k_mono(mp3_path: Path, wav_path: Path) -> Path:
    """
    Converte MP3 para WAV 16kHz mono usando ffmpeg.
    Requisito comum para estabilidade do Whisper.
    """
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(mp3_path),
        "-ac", "1",       # mono
        "-ar", "16000",   # 16k
        str(wav_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"FFmpeg falhou:\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return wav_path
