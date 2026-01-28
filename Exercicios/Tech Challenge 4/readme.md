# Tech Challenge 4 — Monitoramento Multimodal em Saúde da Mulher

## Visão geral
Este repositório contém a solução do Tech Challenge (Fase 4) para monitoramento contínuo de pacientes por meio de dados multimodais (vídeo, áudio e texto), com foco em sinais precoces de risco na saúde e segurança feminina.

O escopo contempla:
- Análise de vídeo clínico com detecção de anomalias e geração de alertas
- Análise de áudio de consultas com extração de sinais vocais e indícios de risco
- Fusão multimodal e pipeline de alerta para a equipe especializada
- Integração com serviços gerenciados em nuvem (Azure), preservando privacidade e segurança

## Problema e objetivos
A rede hospitalar deseja evoluir o uso de IA já aplicada em laudos e exames para também acompanhar continuamente pacientes, usando áudio, vídeo e texto para identificar riscos precoces.

Objetivos atendidos no projeto (seleção):
- Detecção precoce de riscos em saúde materna e ginecológica
- Identificação de sinais de violência doméstica ou abuso
- Monitoramento do bem-estar psicológico feminino
- Uso de serviços em nuvem para ampliar capacidade de processamento
- Detecção de anomalias (preferencialmente em tempo real) para monitoramento preventivo

## Funcionalidades implementadas
### 1) Análise de vídeo especializada
Processamento de vídeos clínicos para identificação de padrões anômalos, com geração de relatórios e alertas.

Inclui:
- Pipeline de extração de frames
- Detecção por modelo YOLOv8 customizado (uma classe/tema definido no projeto)
- Heurísticas de anomalia e sumarização dos achados
- Relatórios automáticos (ex.: desvios em procedimentos, sinais de complicações, desconforto, possíveis casos de violência)

### 2) Análise de áudio especializada
Processamento de áudio de consultas para suporte a sinais de risco (ex.: hesitação, ansiedade, trauma, depressão pós-parto).

Inclui:
- Pré-processamento e normalização do áudio
- Extração de características (energia, pitch, pausas, ritmo) e/ou transcrição (quando aplicável)
- Classificação/score de risco por tipo de consulta e geração de alerta

### 3) Fusão multimodal e alertas
- Unificação de sinais de áudio/vídeo/texto em uma camada de decisão
- Geração de eventos de anomalia e notificação para a equipe médica
- Registro estruturado para auditoria

### 4) Integração Azure
Integração com serviços Azure (por exemplo, Speech/Video/Storage) para suportar processamento e escalabilidade.
A integração é configurável via variáveis de ambiente.

## Arquitetura (alto nível)
1. Ingestão: upload de arquivo (vídeo/áudio) e metadados (texto)
2. Processamento:
   - Vídeo: extração de frames → inferência YOLOv8 → agregação temporal → eventos
   - Áudio: pré-processamento → features/transcrição → score de risco → eventos
3. Fusão e decisão: consolidação de eventos + regras/modelo
4. Saída: relatório + alertas + logs/auditoria

## Como executar
### Pré-requisitos
- Python 3.10+ (recomendado)
- Dependências instaladas no ambiente virtual do projeto
- (Opcional) Credenciais do Azure configuradas via variáveis de ambiente

### Setup
1. Criar e ativar o venv
2. Instalar dependências:
   - `pip install -r requirements.txt`

### Execução (exemplos)
- Pipeline de vídeo: `python -m src.video.run --input data/video.mp4`
- Pipeline de áudio: `python -m src.audio.run --input data/audio.wav`
- Pipeline multimodal: `python -m src.pipeline.run --video ... --audio ... --text ...`

Obs.: ajuste os comandos conforme o entrypoint do seu projeto.

## Estrutura sugerida do repositório
- `src/`
  - `video/` (extração, inferência, pós-processamento, relatórios)
  - `audio/` (pré-processamento, features, transcrição, scoring)
  - `fusion/` (fusão multimodal, regras/modelo)
  - `alerts/` (notificações, persistência, auditoria)
- `models/` (pesos, configs, versionamento)
- `data/` (amostras e exemplos — não versionar dados sensíveis)
- `docs/` (relatório técnico e evidências)
- `notebooks/` (exploração e protótipos)

## Relatório técnico (entrega)
O relatório técnico deve conter:
- Descrição do fluxo multimodal
- Modelos aplicados em cada tipo de dado
- Resultados obtidos e exemplos de anomalias detectadas

O documento pode ficar em `docs/relatorio-tecnico.md` (ou PDF gerado a partir dele).

## Demonstração em vídeo (entrega)
Vídeo de até 15 minutos demonstrando:
- Exemplo prático da análise de áudio e vídeo
- Detecção e resposta a anomalias
- Integração dos serviços Azure
- Fluxo final do alerta à equipe médica

## Privacidade e segurança
- Não versionar dados clínicos reais no repositório
- Anonimizar/mascarar amostras e metadados
- Armazenar segredos apenas em variáveis de ambiente/secret managers
- Registrar apenas o mínimo necessário para auditoria

## Licença
Uso acadêmico conforme orientações do curso/entidade.


Dataset YOLO pronto em: C:\Users\flavio\VsCodeProjects\postech\data\yolo_endovis
YAML: C:\Users\flavio\VsCodeProjects\postech\data\yolo_endovis\dataset.yaml
Treino (exemplo): yolo detect train data=data/yolo_endovis/dataset.yaml model=yolov8n.pt imgsz=640

yolo detect train data=data/yolo_endovis/dataset.yaml model=yolov8n.pt imgsz=640 epochs=10

Results saved to C:\Users\flavio\VsCodeProjects\postech\runs\detect\train
C:\Users\flavio\VsCodeProjects\postech\Exercicios\Tech Challenge 4\scripts\runs\detect\train


video gerado com make_sample_video_from_frames.py
python -m src.video.run --video data/raw/sample.mp4 --weights models/yolo/best.pt --out docs/evidencias-video

pip install azure-storage-blob

Site dos dados - dataset treino
# https://huggingface.co/datasets/tyluan/Endovis2017/blob/main/README.md

rodar script que copia a evidencia do treino
python scripts/collect_training_evidence.py --run-dir "C:\Users\flavio\VsCodeProjects\postech\Exercicios\Tech Challenge 4\runs\detect\runs\yolo_instrument_endovis"





















# Tech Challenge 4 — Análise de Vídeo com YOLOv8 e Integração Azure

## Visão geral
Este repositório implementa um MVP para monitoramento preventivo baseado em análise de vídeo, usando detecção de objetos com YOLOv8 e armazenamento de evidências no Azure Blob Storage.

O projeto foi estruturado para suportar:
- Detecção de instrumentos em frames (classe única: `instrument`)
- Geração de evidências (frames anotados + `events.json`)
- Geração de sinais de risco precoce (baseline por persistência temporal)
- Upload automatizado das evidências para Azure Blob (opcional via flag)

## Objetivos do projeto
- Detecção precoce de riscos (baseline via regras temporais sobre detecções)
- Aplicação em contexto materno/ginecológico (recorte de uso em vídeos clínicos)
- Uso de nuvem (Azure) para armazenamento e escalabilidade

## Dataset (formato YOLO)
O dataset de treino foi preparado no formato YOLO, com:
- `images/train` e `images/val`
- `labels/train` e `labels/val` (bounding boxes em formato YOLO)
- `dataset.yaml` com a configuração do dataset

Estrutura:
data/yolo_endovis/
images/
train/
val/
labels/
train/
val/
dataset.yaml


Formato dos labels:
<class_id> <x_center> <y_center> <width> <height>


No MVP, usamos uma classe:
- `0 = instrument`

## Treinamento do YOLOv8
O treinamento é feito a partir do modelo base `yolov8n.pt` (fine-tuning) e o peso final é copiado para:
- `models/yolo/best.pt`

Exemplo de execução via script:
```powershell
python scripts/train_yolo.py --project runs/detect --name yolo_instrument_endovis --epochs 20 --imgsz 640

python -m src.video.run --video data/raw/sample.mp4 --weights yolov8n.pt --out docs/evidencias-video/compare/pretrained
python -m src.video.run --video data/raw/sample.mp4 --weights models/yolo/best.pt --out docs/evidencias-video/compare/finetuned


(postech-gpu) PS C:\Users\flavio\VsCodeProjects\postech\Exercicios\Tech Challenge 4> python scripts/summarize_events.py --events docs/evidencias-video/compare/pretrained/events.json
File: docs\evidencias-video\compare\pretrained\events.json
detections_count: 12
risk_signals_count: 0
avg_conf: 0.5090
top5:
  t=17.0 conf=0.7481 bbox=[3.22, 240.48, 300.49, 372.86]
  t=16.5 conf=0.6964 bbox=[336.6, 8.08, 507.55, 254.9]
  t=11.0 conf=0.6853 bbox=[4.29, 200.78, 223.17, 334.46]
  t=6.0 conf=0.5663 bbox=[2.93, 128.08, 291.82, 303.88]
  t=20.5 conf=0.5043 bbox=[316.05, 292.63, 509.32, 461.17]

(postech-gpu) PS C:\Users\flavio\VsCodeProjects\postech\Exercicios\Tech Challenge 4> python scripts/summarize_events.py --events docs/evidencias-video/compare/finetuned/events.json
File: docs\evidencias-video\compare\finetuned\events.json
detections_count: 65
risk_signals_count: 1
avg_conf: 0.7626
top5:
  t=8.0 conf=0.9893 bbox=[292.9, 221.42, 507.75, 333.83]
  t=27.0 conf=0.9832 bbox=[9.23, 202.78, 497.31, 510.71]
  t=22.5 conf=0.9732 bbox=[1.91, 110.18, 315.84, 473.1]
  t=15.5 conf=0.9684 bbox=[150.9, 158.25, 507.92, 471.2]
  t=21.0 conf=0.9592 bbox=[3.42, 286.23, 273.96, 506.4]

(postech-gpu) PS C:\Users\flavio\VsCodeProjects\postech\Exercicios\Tech Challenge 4> 


choco install ffmpeg
