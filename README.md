# Pós Tech FIAP - Artificial Intelligence for Devs

Repositório central para os exercícios e desafios técnicos (Tech Challenges) desenvolvidos durante a pós-graduação.

---

## 📂 Projetos

### 1️⃣ Tech Challenge 1: Análise Preditiva de Sobrevivência ao Câncer de Pulmão

Análise exploratória e desenvolvimento de modelo preditivo utilizando dataset de câncer de pulmão do Kaggle, com foco em identificar fatores associados à sobrevivência dos pacientes.

#### 🛠️ Tecnologias Utilizadas
- **Pandas & NumPy**: Manipulação e pré-processamento de dados
- **Scikit-learn**: Modelos de machine learning e métricas
- **Seaborn & Matplotlib**: Visualização de dados e análise exploratória
- **KaggleHub**: Download automatizado do dataset

#### 🧠 Funcionalidades Principais
1. **Pré-processamento**: Encoding de variáveis categóricas (gênero, estágio do câncer, tipo de tratamento, histórico familiar, status de fumante) e feature engineering de duração do tratamento.
2. **Análise Exploratória (EDA)**: Mapas de correlação, distribuições por sobrevivência, análise por país, tipo de tratamento e status de fumo.
3. **Modelagem Preditiva**: Classificação binária de sobrevivência com avaliação de múltiplos algoritmos.

#### 📁 Arquivos
- `TechChallenge_V1.ipynb` — Versão inicial com EDA completa
- `TechChallenge_V2.ipynb` — Versão refinada com modelagem preditiva

---

### 2️⃣ Tech Challenge 2: Classificação de Imagens Médicas com BreastMNIST

Classificação de imagens médicas de ultrassonografia mamária utilizando o dataset BreastMNIST, com exploração de algoritmos evolucionários para otimização de hiperparâmetros.

#### 🛠️ Tecnologias Utilizadas
- **MedMNIST**: Dataset padronizado de imagens médicas (BreastMNIST)
- **Scikit-learn**: Pipelines de classificação e avaliação
- **DEAP**: Algoritmos evolucionários (algoritmos genéticos) para otimização
- **Transformers (Hugging Face)**: Modelos pré-treinados para visão computacional
- **TensorFlow & PyTorch**: Backends para treinamento de redes neurais
- **Seaborn & Matplotlib**: Visualização de resultados

#### 🧠 Funcionalidades Principais
1. **Carregamento e pré-processamento** do dataset BreastMNIST (classificação binária: benigno vs. maligno).
2. **Classificação com ML clássico** e redes neurais convolucionais.
3. **Otimização evolucionária** de hiperparâmetros com DEAP (algoritmos genéticos).
4. **Avaliação de performance** com métricas de classificação médica.

#### 📁 Arquivos
- `TechChallenge_2_v1.ipynb` — Notebook principal com pipeline completo
- `data/breastmnist.npz` — Dataset de imagens médicas
- `requirements.txt` — Dependências do projeto

---

### 3️⃣ Tech Challenge 3: Assistente Médico com IA Generativa

Desenvolvimento de um agente inteligente capaz de auxiliar em diagnósticos e consultas a protocolos médicos, utilizando técnicas de RAG e Fine-Tuning.

#### 🛠️ Tecnologias Utilizadas
- **LangChain & LangGraph**: Para orquestração do agente e gerenciamento de estado.
- **Google Gemini**: Modelo de linguagem principal (`gemini-2.5-flash`).
- **FAISS**: Vector Store para busca semântica (RAG).
- **PyTorch & PEFT (LoRA)**: Para fine-tuning eficiente de modelos.
- **Transformers (Hugging Face)**: Manipulação de modelos e tokenizers.

#### 🧠 Funcionalidades Principais
1. **Agente Orquestrador (`src/agent_orchestrator.py`)**:
   - Decide dinamicamente entre consultar protocolos médicos ou dados de pacientes.
   - Mantém o contexto da conversa.
2. **RAG (Retrieval-Augmented Generation)**:
   - Consulta documentos locais (`data/protocolo_medico_simulado.txt`) para responder perguntas técnicas com precisão.
3. **Fine-Tuning (`src/fine_tunning.py`)**:
   - Script para ajuste fino de modelos (ex: `facebook/opt-125m`) usando dataset customizado (`data/fine_tuning_dataset.jsonl`).

#### 🚀 Como Executar
```bash
cd "Exercicios/Tech Challenge 3"
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Crie .env com: GOOGLE_API_KEY="sua_chave"
python src/agent_orchestrator.py
```

---

### 4️⃣ Tech Challenge 4: Detecção de Instrumentos Cirúrgicos em Vídeos Endoscópicos

Sistema de visão computacional para detecção e rastreamento de instrumentos cirúrgicos em vídeos de cirurgias endoscópicas, utilizando o dataset EndoVis com YOLO.

#### 🛠️ Tecnologias Utilizadas
- **YOLOv8**: Modelo de detecção de objetos em tempo real
- **EndoVis Dataset**: Dataset de cirurgias endoscópicas com anotações de instrumentos
- **OpenCV**: Processamento e análise de vídeo
- **Python Scripts**: Pipeline modular de ingestão, inferência, pós-processamento e relatório

#### 🧠 Funcionalidades Principais
1. **Ingestão (`ingest.py`)**: Pré-processamento e preparação dos frames de vídeo cirúrgico.
2. **Inferência (`infer.py`)**: Detecção de instrumentos cirúrgicos frame a frame com YOLO treinado no EndoVis.
3. **Pós-processamento (`postprocess.py`)**: Filtragem, rastreamento e consolidação das detecções ao longo do vídeo.
4. **Relatório (`report.py`)**: Geração de relatório com estatísticas de presença e frequência dos instrumentos detectados.

#### 📁 Estrutura
```
Tech Challenge 4/
├── scripts/src/video/     # Pipeline de processamento de vídeo
│   ├── ingest.py          # Ingestão de frames
│   ├── infer.py           # Inferência com YOLO
│   ├── postprocess.py     # Pós-processamento
│   └── report.py          # Geração de relatório
└── yolo_endovis/          # Dataset formatado para YOLO
    ├── dataset.yaml       # Configuração (classe: instrument)
    ├── images/            # Frames de treino e validação
    └── labels/            # Anotações YOLO
```

---

### 5️⃣ Tech Challenge 5 — Hackathon: Architecture Analyzer

Sistema back-end baseado em microsserviços para análise automática de diagramas de arquitetura utilizando IA multimodal (Claude). O sistema recebe imagens ou PDFs de diagramas, processa de forma assíncrona e gera relatórios técnicos com componentes identificados, riscos arquiteturais e recomendações.

#### 🛠️ Tecnologias Utilizadas
- **FastAPI**: Framework web para todos os microsserviços
- **Claude Sonnet (Anthropic)**: IA multimodal para análise de diagramas (imagem e PDF)
- **RabbitMQ**: Mensageria assíncrona entre upload e processamento
- **PostgreSQL**: Banco de dados dedicado por serviço
- **Docker Compose**: Orquestração de todos os serviços
- **GitHub Actions**: Pipeline CI/CD (build, testes, deploy)
- **aio_pika & SQLAlchemy**: Comunicação assíncrona com RabbitMQ e banco de dados

#### 🏗️ Arquitetura de Microsserviços

```
[Cliente] → [API Gateway :8000]
                   ↓
          [Upload Service :8001] → [PostgreSQL upload-db]
                   ↓ (RabbitMQ: diagram.process)
        [Processing Service :8002] → [Claude AI]
                   ↓
          [Reports Service :8003] → [PostgreSQL reports-db]
```

| Serviço | Porta | Responsabilidade |
|---------|-------|-----------------|
| API Gateway | 8000 | Ponto de entrada, roteamento, validação |
| Upload Service | 8001 | Recebe arquivo, persiste, publica na fila |
| Processing Service | 8002 | Consome fila, chama Claude AI, salva relatório |
| Reports Service | 8003 | Persiste e expõe relatórios gerados |

#### 🧠 Fluxo da IA
1. PDF ou imagem recebido → salvo em volume compartilhado
2. Mensagem publicada no RabbitMQ (`diagram.process`)
3. Processing Service consome a mensagem e envia o arquivo (base64) ao Claude
4. Claude identifica componentes, riscos e recomendações no diagrama
5. Relatório estruturado (JSON) salvo no Reports Service
6. Status atualizado: `Recebido` → `Em processamento` → `Analisado`

#### 🚀 Como Executar

```bash
cd "Exercicios/Tech Challenge 5/architecture-analyzer"
# Crie o .env com sua chave Anthropic:
echo "ANTHROPIC_API_KEY=sua_chave_aqui" > .env

docker compose up --build
```

Após subir, acesse:
- API Gateway (Swagger): http://localhost:8000/docs
- RabbitMQ Management: http://localhost:15672 (guest/guest)

#### 📡 Endpoints Principais

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `POST` | `/api/v1/analyses` | Upload de diagrama (imagem ou PDF) |
| `GET` | `/api/v1/analyses/{id}` | Consulta status do processamento |
| `GET` | `/api/v1/reports/{id}` | Obtém relatório técnico gerado |

#### 📁 Estrutura
```
architecture-analyzer/
├── api-gateway/           # Roteamento e ponto de entrada
├── upload-service/        # Recepção de arquivos e orquestração
├── processing-service/    # Consumo da fila e análise com IA
├── reports-service/       # Persistência e consulta de relatórios
├── docker-compose.yml     # Orquestração completa
└── .github/workflows/     # CI/CD com GitHub Actions
```
