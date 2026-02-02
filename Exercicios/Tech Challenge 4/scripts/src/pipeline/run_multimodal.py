# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple


SEV_ORDER = {"low": 0, "medium": 1, "high": 2}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def max_severity_from_signals(signals: List[Dict[str, Any]]) -> str:
    sev = "low"
    for s in signals or []:
        cur = s.get("severity", "low")
        if SEV_ORDER.get(cur, 0) > SEV_ORDER.get(sev, 0):
            sev = cur
    return sev


def merge_signals(video_events: Dict[str, Any], audio_events: Dict[str, Any]) -> List[Dict[str, Any]]:
    merged = []

    for s in video_events.get("risk_signals", []) or []:
        merged.append({
            "modality": "video",
            "type": s.get("type"),
            "severity": s.get("severity", "low"),
            "reason": s.get("reason", ""),
            "t_start": s.get("t_start"),
            "t_end": s.get("t_end"),
        })

    for s in audio_events.get("risk_signals", []) or []:
        merged.append({
            "modality": "audio",
            "type": s.get("type"),
            "severity": s.get("severity", "low"),
            "reason": s.get("reason", ""),
        })

    # ordena por severidade (high primeiro)
    merged.sort(key=lambda x: SEV_ORDER.get(x.get("severity", "low"), 0), reverse=True)
    return merged


def build_alerts(multimodal_events: Dict[str, Any]) -> Dict[str, Any]:
    signals = multimodal_events.get("risk_signals", []) or []
    final_sev = multimodal_events.get("summary", {}).get("max_severity", "low")

    # alerta simples e direto (pode evoluir)
    alert = {
        "run_id": multimodal_events["run_id"],
        "created_at": multimodal_events["created_at"],
        "severity": final_sev,
        "action": "review_required" if final_sev in ("medium", "high") else "monitor",
        "reasons": [s.get("reason", "") for s in signals[:6] if s.get("reason")],
        "signals_count": len(signals),
        "sources": multimodal_events.get("sources", {}),
    }

    return {
        "alerts": [alert],
        "summary": {
            "alerts_count": 1,
            "max_severity": final_sev,
        }
    }


def build_report(multimodal_events: Dict[str, Any], alerts: Dict[str, Any]) -> str:
    sources = multimodal_events.get("sources", {})
    v = sources.get("video", {})
    a = sources.get("audio", {})

    lines = []
    lines.append("# Evidências — Multimodal (Vídeo + Áudio)")
    lines.append("")
    lines.append(f"- Run ID (UTC): `{multimodal_events['run_id']}`")
    lines.append(f"- Gerado em (UTC): {multimodal_events['created_at']}")
    lines.append("")
    lines.append("## Fontes")
    lines.append(f"- Vídeo events: `{v.get('events_path', '')}` | run_id: `{v.get('run_id', '')}` | video_id: `{v.get('video_id', '')}`")
    lines.append(f"- Áudio events: `{a.get('events_path', '')}` | run_id: `{a.get('run_id', '')}` | audio_id: `{a.get('audio_id', '')}`")
    lines.append("")
    lines.append("## Sumário")
    lines.append(f"- Sinais combinados: **{multimodal_events['summary']['risk_signals_count']}**")
    lines.append(f"- Severidade máxima: **{multimodal_events['summary']['max_severity']}**")
    lines.append(f"- Ação recomendada: **{alerts['alerts'][0]['action']}**")
    lines.append("")
    lines.append("## Sinais (ordenados por severidade)")
    if not multimodal_events.get("risk_signals"):
        lines.append("- Nenhum sinal identificado.")
    else:
        for s in multimodal_events["risk_signals"][:25]:
            mod = s.get("modality", "?")
            sev = s.get("severity", "low")
            typ = s.get("type", "")
            reason = s.get("reason", "")
            t0 = s.get("t_start")
            t1 = s.get("t_end")
            if mod == "video" and t0 is not None and t1 is not None:
                lines.append(f"- **{sev}** | {mod} | {typ} | {reason} | {t0}s → {t1}s")
            else:
                lines.append(f"- **{sev}** | {mod} | {typ} | {reason}")
    lines.append("")
    lines.append("## Observações")
    lines.append("- Fusão multimodal por regra simples: severidade final = máximo entre sinais de áudio e vídeo.")
    lines.append("- Saída inclui `alerts.json` para consumo por fluxo de resposta clínica/operacional.")
    lines.append("")
    return "\n".join(lines)


def main(
    video_events_path: str = "docs/evidencias-video/events.json",
    audio_events_path: str = "docs/evidencias-audio/events.json",
    out_dir: str = "docs/evidencias-multimodal",
    upload_azure: bool = False,
):
    video_events_path = Path(video_events_path)
    audio_events_path = Path(audio_events_path)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    video_events = load_json(video_events_path)
    audio_events = load_json(audio_events_path)

    run_id = utc_run_id()

    merged_signals = merge_signals(video_events, audio_events)
    max_sev = max_severity_from_signals(merged_signals)

    multimodal_events = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sources": {
            "video": {
                "events_path": str(video_events_path).replace("\\", "/"),
                "run_id": video_events.get("run_id", ""),
                "video_id": video_events.get("video_id", ""),
            },
            "audio": {
                "events_path": str(audio_events_path).replace("\\", "/"),
                "run_id": audio_events.get("run_id", ""),
                "audio_id": audio_events.get("audio_id", audio_events.get("source", {}).get("uri", "")),
            },
        },
        "risk_signals": merged_signals,
        "summary": {
            "risk_signals_count": len(merged_signals),
            "max_severity": max_sev,
        },
    }

    alerts = build_alerts(multimodal_events)

    out_events = out_dir / "events.json"
    out_alerts = out_dir / "alerts.json"
    out_report = out_dir / "report.md"

    out_events.write_text(json.dumps(multimodal_events, ensure_ascii=False, indent=2), encoding="utf-8")
    out_alerts.write_text(json.dumps(alerts, ensure_ascii=False, indent=2), encoding="utf-8")
    out_report.write_text(build_report(multimodal_events, alerts), encoding="utf-8")

    print(f"OK: {out_events}")
    print(f"OK: {out_alerts}")
    print(f"OK: {out_report}")

    if upload_azure:
        from src.azure.storage import upload_file

        base_prefix = f"runs/{run_id}/multimodal"
        upload_file(str(out_events), f"{base_prefix}/events.json")
        upload_file(str(out_alerts), f"{base_prefix}/alerts.json")
        upload_file(str(out_report), f"{base_prefix}/report.md")
        print(f"OK: uploaded to Azure prefix: {base_prefix}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-events", default="docs/evidencias-video/events.json")
    ap.add_argument("--audio-events", default="docs/evidencias-audio/events.json")
    ap.add_argument("--out", default="docs/evidencias-multimodal")
    ap.add_argument("--upload-azure", action="store_true")
    args = ap.parse_args()

    main(
        video_events_path=args.video_events,
        audio_events_path=args.audio_events,
        out_dir=args.out,
        upload_azure=args.upload_azure,
    )
