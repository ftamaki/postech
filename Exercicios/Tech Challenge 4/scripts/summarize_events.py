import json
from pathlib import Path
from statistics import mean

def main(path):
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    dets = data.get("detections", [])
    risks = data.get("risk_signals", [])

    confs = [d["conf"] for d in dets if "conf" in d]
    avg_conf = mean(confs) if confs else 0.0

    # top 5 detecções por confiança
    top = sorted(dets, key=lambda x: x.get("conf", 0), reverse=True)[:5]

    print(f"File: {p}")
    print(f"detections_count: {data.get('summary', {}).get('detections_count', len(dets))}")
    print(f"risk_signals_count: {data.get('summary', {}).get('risk_signals_count', len(risks))}")
    print(f"avg_conf: {avg_conf:.4f}")
    print("top5:")
    for t in top:
        print(f"  t={t['t']} conf={t['conf']} bbox={t['bbox_xyxy']}")
    print()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    args = ap.parse_args()
    main(args.events)
