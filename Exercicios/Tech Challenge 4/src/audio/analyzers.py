# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone

from .rules import RISK_RULES_PTBR, BOOST_TERMS


def _normalize_text(s: str) -> str:
    return (s or "").lower().strip()


def _count_hits(text: str, keywords: List[str]) -> List[str]:
    t = _normalize_text(text)
    hits = []
    for k in keywords:
        if _normalize_text(k) in t:
            hits.append(k)
    return hits


def _has_any_boost(text: str) -> bool:
    t = _normalize_text(text)
    return any(bt in t for bt in BOOST_TERMS)


def _severity_max(a: str, b: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    return a if order.get(a, 0) >= order.get(b, 0) else b


def build_risk_signals(transcript: str) -> List[Dict[str, Any]]:
    """
    Baseline: regras por keywords no transcript.
    Retorna risk_signals no formato consistente com seu pipeline de vídeo.
    """
    signals = []
    if not transcript:
        return signals

    for rule in RISK_RULES_PTBR:
        hits = _count_hits(transcript, rule["keywords_any"])
        if hits:
            sev = rule["severity"]
            if _has_any_boost(transcript) and sev == "medium":
                sev = "high"  # boost simples
            signals.append({
                "type": rule["id"],
                "severity": sev,
                "reason": f"{rule['label']} (keywords: {', '.join(hits[:6])}{'...' if len(hits) > 6 else ''})",
            })

    return signals


def summarize_transcript(transcript: str) -> Dict[str, Any]:
    """
    Métricas simples para auditoria (sem NLP pesado).
    """
    t = _normalize_text(transcript)
    words = [w for w in t.replace("\n", " ").split(" ") if w]
    word_count = len(words)
    char_count = len(transcript or "")
    return {
        "word_count": word_count,
        "char_count": char_count,
    }


def max_severity(signals: List[Dict[str, Any]]) -> str:
    sev = "low"
    for s in signals or []:
        sev = _severity_max(sev, s.get("severity", "low"))
    return sev


def build_events(
    audio_path: str,
    model_name: str,
    language: str,
    transcript: str,
    segments: List[Dict[str, Any]],
    run_id: str,
) -> Dict[str, Any]:
    signals = build_risk_signals(transcript)
    summary = summarize_transcript(transcript)
    summary["risk_signals_count"] = len(signals)
    summary["max_severity"] = max_severity(signals)

    events = {
        "run_id": run_id,
        "audio_id": audio_path,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {"type": "local", "uri": audio_path},
        "model": {"name": "whisper-local", "variant": model_name, "language": language},
        "transcript": transcript,
        "segments": segments,  # [{start, end, text}]
        "risk_signals": signals,
        "summary": summary,
    }
    return events


def build_report(events: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Evidências — Análise de Áudio (Whisper local)")
    lines.append("")
    lines.append(f"- Run ID (UTC): `{events.get('run_id')}`")
    lines.append(f"- Gerado em (UTC): {events.get('created_at')}")
    lines.append(f"- Áudio: `{events.get('audio_id')}`")
    lines.append(f"- Modelo: whisper-local `{events['model']['variant']}` | language={events['model']['language']}")
    lines.append("")
    lines.append("## Sumário")
    lines.append(f"- Palavras: **{events['summary'].get('word_count', 0)}**")
    lines.append(f"- Sinais de risco: **{events['summary'].get('risk_signals_count', 0)}**")
    lines.append(f"- Severidade máxima: **{events['summary'].get('max_severity', 'low')}**")
    lines.append("")
    lines.append("## Sinais de risco (baseline)")
    if not events.get("risk_signals"):
        lines.append("- Nenhum sinal identificado pelas regras atuais.")
    else:
        for rs in events["risk_signals"]:
            lines.append(f"- **{rs['severity']}** | {rs['type']} | {rs['reason']}")
    lines.append("")
    lines.append("## Transcrição (primeiros 600 caracteres)")
    txt = (events.get("transcript") or "").strip()
    lines.append("```")
    lines.append(txt[:600] + ("..." if len(txt) > 600 else ""))
    lines.append("```")
    lines.append("")
    lines.append("## Observações")
    lines.append("- Este módulo usa transcrição local (Whisper) e regras explicáveis por palavras-chave.")
    lines.append("- Pode evoluir para classificação supervisionada e sinais acústicos mais robustos.")
    lines.append("")
    return "\n".join(lines)
