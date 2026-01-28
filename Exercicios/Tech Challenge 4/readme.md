# Tech Challenge 4 — Monitoramento Multimodal em Saúde da Mulher (Vídeo + Áudio + Azure)

## Visão geral
Este repositório contém a solução do Tech Challenge (Fase 4) para monitoramento contínuo de pacientes por meio de dados multimodais (vídeo e áudio), com foco em sinais precoces de risco na saúde e segurança feminina.

O escopo contempla:
- Análise de vídeo com detecção especializada (YOLOv8 fine-tuned) e geração de evidências
- Análise de áudio (MP3) com transcrição local (Whisper) e triagem por regras/keywords
- Fusão multimodal e geração de alerta consolidado (`alerts.json`)
- Integração com Azure Blob Storage para armazenamento de evidências por execução (`run_id`)

---

## Problema e objetivos
A rede hospitalar deseja evoluir o uso de IA já aplicada em laudos e exames para também acompanhar continuamente pacientes, usando dados multimodais para identificar riscos precoces.

Objetivos atendidos no projeto:
- Detecção precoce de riscos em saúde materna e ginecológica (baseline explicável)
- Triagem de sinais associados a violência doméstica/abuso (baseline via transcrição + regras)
- Monitoramento de bem-estar psicológico (baseline via transcrição + regras)
- Uso de serviços em nuvem (Azure Blob) para evidências e rastreabilidade
- Detecção de anomalias (MVP) e geração de alertas para equipe especializada

---

## Funcionalidades implementadas

### 1) Análise de vídeo (YOLOv8 fine-tuned)
- Inferência YOLOv8 para detecção de instrumento (classe única: `instrument`)
- Geração de evidências:
  - `events.json`
  - frames anotados (`frames_annotated/`)
  - `report.md`
- Regra de “detecção precoce” (MVP): persistência temporal do instrumento gera `risk_signals`

### 2) Análise de áudio (MP3 + Whisper local)
- Normalização com FFmpeg (MP3 → WAV 16k mono)
- Transcrição local com Whisper
- Geração de `risk_signals` por regras/keywords (baseline explicável)
- Evidências:
  - `events.json`
  - `report.md`

### 3) Fusão multimodal e alertas
- Consolidação dos `risk_signals` de vídeo e áudio
- Severidade final = máximo entre severidades (MVP)
- Geração de `alerts.json` com ação recomendada (`monitor` ou `review_required`)
- Evidências multimodais:
  - `events.json`
  - `alerts.json`
  - `report.md`

### 4) Integração Azure (Blob Storage)
- Upload opcional das evidências para Azure Blob
- Organização por execução:
  - `runs/<run_id>/video/...`
  - `runs/<run_id>/audio/...`
  - `runs/<run_id>/multimodal/...`

---

## Arquitetura (alto nível)
1. Ingestão: vídeo (MP4) e áudio (MP3)
2. Processamento:
   - Vídeo: extração/amostragem de frames → inferência YOLOv8 → agregação temporal → eventos + evidências
   - Áudio: normalização → transcrição → regras/score → eventos + evidências
3. Fusão e decisão: consolidação multimodal → alerta final
4. Saída: relatórios (`report.md`), eventos (`events.json`) e alertas (`alerts.json`) + evidências para auditoria

---

## Estrutura do repositório (alto nível)
```text
src/
  azure/
    storage.py
  video/
    run.py
  audio/
    run.py
    analyzers.py
    rules.py
    utils.py
  pipeline/
    run_multimodal.py
scripts/
  train_yolo.py
  collect_training_evidence.py
docs/
  relatorio_tecnico.md
  compare_pretrained_vs_finetuned.md
  evidencias-video/
  evidencias-audio/
  evidencias-multimodal/
models/
  yolo/
data/
  yolo_endovis/
  raw/
```

---

## Requisitos

### Ambiente
- Python 3.10+ (recomendado)
- Ambiente virtual (venv/conda) ativo

### Dependências principais
- `ultralytics`
- `opencv-python`
- `openai-whisper`
- `torch` (GPU opcional)
- `azure-storage-blob` (para upload de evidências)

Instalação:
```powershell
pip install -r requirements.txt
```

### FFmpeg (necessário para MP3 no módulo de áudio)
Verificação:
```powershell
ffmpeg -version
```

Instalação (exemplos):
- Chocolatey:
```powershell
choco install ffmpeg
```

---

## Configuração do Azure (Blob Storage)
Configurar variáveis de ambiente (ex.: `.env` ou export na sessão):
- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_CONTAINER_NAME`

Validação:
```powershell
python -c "import os; print(bool(os.getenv('AZURE_STORAGE_CONNECTION_STRING'))); print(os.getenv('AZURE_CONTAINER_NAME'))"
```

---

## Dataset (YOLO)
Dataset preparado no formato YOLO em:
- `data/yolo_endovis/`
- `data/yolo_endovis/dataset.yaml`

Fonte (dataset de referência):
- https://huggingface.co/datasets/tyluan/Endovis2017/blob/main/README.md

---

## Como executar

### 1) Treinamento YOLO (fine-tuning)
Treina e publica o peso final em `models/yolo/best.pt`:
```powershell
python scripts/train_yolo.py --epochs 20 --imgsz 640 --batch 16 --project . --name yolo_instrument_endovis
```

Evidência do run em:
- `runs/detect/yolo_instrument_endovis/`

Copiar evidências do treino para `docs/`:
```powershell
python scripts/collect_training_evidence.py --run-dir ".\runs\detect\yolo_instrument_endovis" --out-dir ".\docs\evidencias-video\treino\yolo_instrument_endovis"
```

### 2) Vídeo — inferência e evidências
Rodar local:
```powershell
python -m src.video.run --video data/raw/sample.mp4 --weights models/yolo/best.pt --out docs/evidencias-video
```

Rodar e enviar para Azure:
```powershell
python -m src.video.run --video data/raw/sample.mp4 --weights models/yolo/best.pt --out docs/evidencias-video --upload-azure
```

### 3) Áudio — inferência e evidências (MP3)
Rodar local:
```powershell
python -m src.audio.run --audio data/raw/sample.mp3 --model small --lang pt --out docs/evidencias-audio
```

Rodar e enviar para Azure:
```powershell
python -m src.audio.run --audio data/raw/sample.mp3 --model small --lang pt --out docs/evidencias-audio --upload-azure
```

### 4) Multimodal — fusão e alerta final
Rodar local:
```powershell
python -m src.pipeline.run_multimodal
```

Rodar e enviar para Azure:
```powershell
python -m src.pipeline.run_multimodal --upload-azure
```

---

## Evidências e resultados
- Treino YOLO:
  - `docs/evidencias-video/treino/yolo_instrument_endovis/`
- Comparação pré-treinado vs fine-tuned:
  - `docs/evidencias-video/compare/pretrained/`
  - `docs/evidencias-video/compare/finetuned/`
  - `docs/compare_pretrained_vs_finetuned.md`
- Multimodal:
  - `docs/evidencias-multimodal/` (events + alerts + report)

---

## Relatório técnico (entrega)
O relatório técnico está em:
- `docs/relatorio_tecnico.md`

Conteúdo:
- Fluxo multimodal
- Modelos e estratégias por modalidade
- Evidências e exemplos gerados (`events.json`, `report.md`, `alerts.json`)
- Integração Azure e organização por `run_id`

---

## Privacidade e segurança
- Não versionar dados clínicos reais no repositório
- Anonimizar/mascarar amostras e metadados
- Armazenar segredos apenas via variáveis de ambiente / secret manager
- Registrar apenas o mínimo necessário para auditoria

---

## Licença
Uso acadêmico conforme orientações do curso/entidade.
