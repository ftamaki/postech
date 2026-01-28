# Semana 1 — Resumo (Dataset YOLO + Treino + Pipeline de Evidências)

Este documento resume **o que foi feito na Semana 1** e **como o YOLO treina** usando o dataset preparado para o projeto.

---

## 1) Decisões do MVP

- **Funcionalidades do desafio**
  - Análise de **vídeos** com detecção de objetos via **YOLOv8**.
  - **Integração com Azure** (Blob Storage) para armazenar evidências do processamento.

- **Objetivos (3)**
  - **Detecção precoce de riscos** (baseline via regras temporais sobre detecções).
  - **Riscos maternos/ginecológicos** (recorte clínico do caso de uso).
  - **Uso de nuvem (Azure)** para armazenamento e escalabilidade.

---

## 2) O que é “Dataset YOLO” (no nosso contexto)

Um **dataset YOLO** é um dataset organizado para treino de detecção no padrão YOLO:

```
data/yolo_endovis/
  images/
    train/  (imagens)
    val/
  labels/
    train/  (arquivos .txt com bounding boxes)
    val/
  dataset.yaml
```

### Labels (formato YOLO)
Para cada imagem `images/train/xxx.jpg`, existe um arquivo de mesmo nome em `labels/train/xxx.txt`.

Cada linha do `.txt` é uma bounding box:

```
<class_id> <x_center> <y_center> <width> <height>
```

- Valores **normalizados (0–1)** em relação ao tamanho da imagem.
- No MVP, usamos **uma classe**:
  - `0 = instrument`

Exemplo:

```
0 0.512300 0.433200 0.220000 0.180000
```

---

## 3) Como geramos o dataset YOLO (EndoVis → YOLO)

O EndoVis (usado como base prática do MVP) fornece **imagem + máscara** (segmentação).  
YOLO precisa de **bounding box** (caixa).

Fizemos a conversão:

**máscara (pixels do instrumento) → contorno → bounding box → label YOLO**

Resultado:
- `data/yolo_endovis/images/...` com frames
- `data/yolo_endovis/labels/...` com caixas no padrão YOLO
- `data/yolo_endovis/dataset.yaml` para o `yolo detect train`

---

## 4) Como o YOLO treina (resumo)

### Entrada do treino
Para cada imagem do treino, o YOLO lê o `.txt` correspondente e recebe as caixas “verdadeiras” (ground truth).

### O que o modelo prevê
O modelo tenta prever várias caixas candidatas por imagem:
- posição e tamanho da caixa (x, y, w, h)
- confiança (objectness)
- classe (aqui: `instrument`)

### Como ele aprende (loss)
Ele compara a previsão com os rótulos e ajusta os pesos para reduzir:
- **erro de localização** (caixa prevista vs caixa real — baseado em IoU/CIoU)
- **erro de confiança** (prever objeto onde não tem / deixar de prever onde tem)
- **erro de classe** (aqui simplificado: só existe uma classe)

Isso se repete por várias **epochs** (passagens completas no dataset).

### Exemplo aplicado ao nosso dataset
- Imagem: `data/yolo_endovis/images/train/endovis_train_000123.jpg`
- Label:  `data/yolo_endovis/labels/train/endovis_train_000123.txt`

Se o label contém:
```
0 0.48 0.52 0.30 0.25
```
o modelo aprende que existe um **instrumento** nessa região. No começo ele erra, mas com epochs ele ajusta:
- melhorar o encaixe da caixa
- aumentar confiança quando há instrumento
- reduzir falsos positivos

---

## 5) Pipeline de vídeo (inferência → evidências)

Implementamos um pipeline MVP que:
1. Lê um vídeo via OpenCV (`cv2.VideoCapture`)
2. Amostra frames a cada `FRAME_STRIDE` (para reduzir custo)
3. Roda inferência YOLOv8 (`model.predict`)
4. Salva evidências:
   - frames anotados em `docs/evidencias-video/frames_annotated/`
   - `docs/evidencias-video/events.json` com detecções e metadados

### Detecção precoce de riscos (baseline)
Geramos `risk_signals` com uma regra simples:
- registra timestamps com detecção
- agrupa em janelas contínuas
- se o objeto persistir por ≥ `RISK_PERSISTENCE_SEC`, gera risco `medium`
- se persistir por ≥ 20s, gera risco `high`

---

## 6) Integração com Azure (Blob Storage)

Adicionamos módulo de Azure para:
- garantir container
- upload de evidências (`events.json` e frames anotados)

O pipeline suporta flag `--upload-azure` para enviar os artefatos ao Blob Storage.

---

## 7) Resultado final da Semana 1 (checklist)

- [x] Dataset YOLO gerado (imagens + labels + dataset.yaml)
- [x] Treino YOLOv8 baseline (peso `best.pt` gerado)
- [x] Pipeline de inferência em vídeo (frames anotados + `events.json`)
- [x] Upload para Azure Blob funcionando

---

## Comandos (exemplo)

Treino:
```powershell
yolo detect train data=data/yolo_endovis/dataset.yaml model=yolov8n.pt imgsz=640 epochs=10
```

Inferência + evidências:
```powershell
python -m src.video.run --video data/raw/sample.mp4 --weights models/yolo/best.pt --out docs/evidencias-video --upload-azure
```
