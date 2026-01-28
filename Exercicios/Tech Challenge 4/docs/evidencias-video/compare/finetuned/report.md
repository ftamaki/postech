# Evidências — Análise de Vídeo (YOLOv8)

- Vídeo: `sample.mp4`
- Run ID (UTC): `2026-01-27_012841Z`
- Gerado em (UTC): 2026-01-27T01:28:41.804921Z
- Modelo: yolov8 | weights: `models/yolo/best.pt` | conf_th=0.35
- FPS: 10.00 | stride: 5 | frames: 300

## Sumário
- Detecções: **65**
- Sinais de risco: **1**
- Severidade máxima: **medium**

## Sinais de risco (detecção precoce)
- **medium** | Persistência de instrument por ~17.0s | 0.0s → 17.0s

## Observações técnicas
- Relatório gerado por um MVP com regras simples de persistência temporal.
- A lógica pode evoluir para modelos temporais/etapas clínicas e thresholds calibrados.
