# Evidências — Multimodal (Vídeo + Áudio)

- Run ID (UTC): `2026-01-28_013511Z`
- Gerado em (UTC): 2026-01-28T01:35:11.415472Z

## Fontes
- Vídeo events: `docs/evidencias-video/events.json` | run_id: `2026-01-27_021547Z` | video_id: `sample.mp4`
- Áudio events: `docs/evidencias-audio/events.json` | run_id: `2026-01-28_010910Z` | audio_id: `sample.mp3`

## Sumário
- Sinais combinados: **5**
- Severidade máxima: **high**
- Ação recomendada: **review_required**

## Sinais (ordenados por severidade)
- **high** | audio | anxiety_signal | Sinal de ansiedade (keywords: ansiosa, nervosa, apreensiva, preocupada, falta de ar, medo)
- **high** | audio | depression_signal | Sinal de depressão / humor deprimido (keywords: triste, muito triste, sem esperança, desanimada, não aguento, culpa...)
- **high** | audio | postpartum_signal | Sinal de risco pós-parto (triagem textual) (keywords: depois que o bebê nasceu, não consigo dormir, me sinto incapaz)
- **high** | audio | domestic_violence_signal | Sinal de possível violência doméstica (keywords: ameaça, tenho medo dele, ele me controla)
- **medium** | video | early_risk | Persistência de instrument por ~17.0s | 0.0s → 17.0s

## Observações
- Fusão multimodal por regra simples: severidade final = máximo entre sinais de áudio e vídeo.
- Saída inclui `alerts.json` para consumo por fluxo de resposta clínica/operacional.
