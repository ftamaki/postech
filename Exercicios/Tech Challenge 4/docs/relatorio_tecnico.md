# Relatório Técnico — Tech Challenge 4 (Multimodal: Vídeo + Áudio + Azure)

## 1. Objetivo
Este projeto implementa um pipeline multimodal para detecção precoce de sinais de risco em contexto materno/ginecológico, combinando:
- Visão computacional (vídeo) para detecção de instrumento (YOLOv8 fine-tuned)
- Processamento de áudio (MP3) com transcrição (Whisper local) e regras baseline para triagem de risco
- Fusão multimodal (vídeo + áudio) para geração de um alerta consolidado
- Integração com Azure Blob Storage para armazenamento de evidências por execução (run_id)

O foco é entregar um MVP reprodutível, com evidências (artefatos) e outputs explicáveis.

---

## 2. Arquitetura da solução

### 2.1 Fluxo local (offline)
1) Treino do YOLOv8 (fine-tuning) e publicação do `best.pt`
2) Inferência em vídeo (`src/video/run.py`) gerando:
   - `docs/evidencias-video/events.json`
   - `docs/evidencias-video/report.md`
   - `docs/evidencias-video/frames_annotated/*.jpg`
3) Inferência em áudio MP3 (`src/audio/run.py`) gerando:
   - `docs/evidencias-audio/events.json`
   - `docs/evidencias-audio/report.md`
4) Fusão multimodal (`src/pipeline/run_multimodal.py`) gerando:
   - `docs/evidencias-multimodal/events.json`
   - `docs/evidencias-multimodal/alerts.json`
   - `docs/evidencias-multimodal/report.md`

### 2.2 Fluxo Azure (armazenamento de evidências)
Os módulos enviam evidências para Azure Blob Storage no padrão:
- `runs/<run_id>/video/<video_name>/...`
- `runs/<run_id>/audio/<audio_name>/...`
- `runs/<run_id>/multimodal/...`

Isso garante:
- rastreabilidade por execução
- não sobrescrever evidências
- auditoria de inputs/outputs

---

## 3. Módulo de Vídeo (YOLOv8 fine-tuned)

### 3.1 Dataset e formato
O dataset foi preparado no formato YOLO (imagens + labels com bounding boxes):
- `data/yolo_endovis/images/{train,val}/`
- `data/yolo_endovis/labels/{train,val}/`
- `data/yolo_endovis/dataset.yaml`

Formato de cada label:
`<class_id> <x_center> <y_center> <width> <height>`

Neste MVP, há uma classe alvo:
- `0 = instrument`

### 3.2 Treinamento
O treinamento é executado via:
- `scripts/train_yolo.py`

Output do run:
- `runs/detect/yolo_instrument_endovis/`

Artefato publicado para uso no pipeline:
- `models/yolo/best.pt`

### 3.3 Inferência e evidências
O pipeline de inferência:
- lê o vídeo (OpenCV)
- aplica YOLOv8 com `conf_th`
- amostra frames com `frame_stride` (performance)
- salva evidências:
  - `events.json` com detecções e metadados
  - frames anotados (amostra) em `frames_annotated/`

### 3.4 Detecção precoce (baseline)
Uma regra simples gera `risk_signals` se houver persistência do instrumento por uma janela de tempo:
- `medium`: persistência >= `RISK_PERSISTENCE_SEC`
- `high`: persistência >= 20s

Esse baseline é explicável, rápido e serve como ponto de partida.

### 3.5 Comparação pré-treinado vs fine-tuned
Foi executada comparação entre:
- modelo pré-treinado (genérico)
- modelo fine-tuned no dataset do domínio

A comparação e evidências estão registradas em:
- `docs/evidencias-video/compare/pretrained/`
- `docs/evidencias-video/compare/finetuned/`
- `docs/compare_pretrained_vs_finetuned.md`

Resultado observado: melhora no desempenho do fine-tuned no domínio (ex.: maior consistência/aderência/qualidade das detecções).

---

## 4. Módulo de Áudio (MP3 + Whisper local)

### 4.1 Objetivo
Processar áudio em MP3 (ex.: fala/consulta) para:
- transcrever o conteúdo
- extrair sinais de risco via regras baseline (triagem)
- gerar evidências (`events.json`, `report.md`)
- opcionalmente enviar tudo ao Azure

### 4.2 Pipeline
1) Normalização do MP3 com FFmpeg (conversão para WAV 16kHz mono)
2) Transcrição local via Whisper (`openai-whisper`)
3) Regras por palavras-chave para `risk_signals` (baseline explicável)

Arquivos:
- `src/audio/run.py`
- `src/audio/analyzers.py`
- `src/audio/rules.py`
- `src/audio/utils.py`

### 4.3 Regras baseline (keywords)
Os sinais de risco são gerados por match de keywords em PT-BR, exemplo:
- ansiedade
- humor deprimido / depressão
- risco pós-parto (triagem textual)
- possível violência doméstica

O output é padronizado com `risk_signals` e severidade (`low|medium|high`).

---

## 5. Fusão Multimodal e Alerta Final

### 5.1 Objetivo
Unificar os sinais de risco de vídeo e áudio para gerar:
- `events.json` multimodal
- `alerts.json` (alerta consolidado para resposta)

### 5.2 Regra de fusão (MVP)
- `risk_signals` final = concatenação dos sinais das modalidades
- severidade final (`max_severity`) = maior severidade entre os sinais
- `alerts.json` inclui:
  - severidade
  - ação recomendada (`monitor` ou `review_required`)
  - razões principais (top sinais)

Script:
- `src/pipeline/run_multimodal.py`

---

## 6. Integração com Azure Blob Storage

### 6.1 Configuração
O projeto utiliza variáveis de ambiente:
- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_CONTAINER_NAME`

### 6.2 Organização no Blob
Saída organizada por `run_id`:
- `runs/<run_id>/video/...`
- `runs/<run_id>/audio/...`
- `runs/<run_id>/multimodal/...`

Isso garante rastreabilidade e evita sobrescrita.

---

## 7. Como executar (reprodutibilidade)

### 7.1 Treinar YOLO
    ```powershell
    python scripts/train_yolo.py --epochs 20 --imgsz 640 --batch 16 --project . --name yolo_instrument_endovis

### 7.2 Rodar inferência de vídeo (local + Azure opcional)
    python -m src.video.run --video data/raw/sample.mp4 --weights models/yolo/best.pt --out docs/evidencias-video
    python -m src.video.run --video data/raw/sample.mp4 --weights models/yolo/best.pt --out docs/evidencias-video --upload-azure

### 7.3 Rodar inferência de áudio (MP3) (local + Azure opcional)
    python -m src.audio.run --audio data/raw/sample.mp3 --model small --lang pt --out docs/evidencias-audio
    python -m src.audio.run --audio data/raw/sample.mp3 --model small --lang pt --out docs/evidencias-audio --upload-azure

### 7.4 Rodar fusão multimodal (local + Azure opcional)
    python -m src.pipeline.run_multimodal
    python -m src.pipeline.run_multimodal --upload-azure

### 8. Evidências e artefatos gerados
### 8.1 Vídeo

- docs/evidencias-video/events.json
- docs/evidencias-video/report.md
- docs/evidencias-video/frames_annotated/

### 8.2 Áudio

- docs/evidencias-audio/events.json
- docs/evidencias-audio/report.md

### 8.3 Multimodal

- docs/evidencias-multimodal/events.json
- docs/evidencias-multimodal/alerts.json
- docs/evidencias-multimodal/report.md


---
## Requisitos
- Python (venv ativo)
- FFmpeg no PATH (necessário para MP3 no módulo de áudio)
- Dependências Python:
  - ultralytics
  - opencv-python
  - openai-whisper
  - torch (preferencialmente com CUDA para GPU)

Verificação do ffmpeg:
```powershell
ffmpeg -version
