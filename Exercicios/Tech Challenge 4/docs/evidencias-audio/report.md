# Evidências — Análise de Áudio (Whisper local)

- Run ID (UTC): `2026-01-28_010910Z`
- Gerado em (UTC): 2026-01-28T01:09:16.665228Z
- Áudio: `sample.mp3`
- Modelo: whisper-local `small` | language=pt

## Sumário
- Palavras: **89**
- Sinais de risco: **4**
- Severidade máxima: **high**

## Sinais de risco (baseline)
- **high** | anxiety_signal | Sinal de ansiedade (keywords: ansiosa, nervosa, apreensiva, preocupada, falta de ar, medo)
- **high** | depression_signal | Sinal de depressão / humor deprimido (keywords: triste, muito triste, sem esperança, desanimada, não aguento, culpa...)
- **high** | postpartum_signal | Sinal de risco pós-parto (triagem textual) (keywords: depois que o bebê nasceu, não consigo dormir, me sinto incapaz)
- **high** | domestic_violence_signal | Sinal de possível violência doméstica (keywords: ameaça, tenho medo dele, ele me controla)

## Transcrição (primeiros 600 caracteres)
```
Olá, doutora. Eu estou muito ansiosa, nervosa e preocupada. Tenho medo, fico apreensiva e às vezes sinto falta de ar e taque cardia. Depois que o bebê nasceu, eu não consigo dormir e estou exausta. Eu me sinto muito triste, desanimada e sem esperança. Eu chore e me sinto culpada, como se eu não aguento mais. Eu também me sinto incapaz de cuidar do bebê e não consigo criar vínculo. Está intenso e difícil todos os dias. Em casa, tenho medo dele. Ele me controla, grita e faz ameaças.
```

## Observações
- Este módulo usa transcrição local (Whisper) e regras explicáveis por palavras-chave.
- Pode evoluir para classificação supervisionada e sinais acústicos mais robustos.


python -m src.audio.run --audio data/raw/sample.mp3 --model small --lang pt --out docs/evidencias-audio --upload-azure

