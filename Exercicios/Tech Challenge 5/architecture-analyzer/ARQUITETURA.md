# Architecture Analyzer — Documentação dos Serviços

## Visão Geral

O sistema é composto por **4 microserviços Python + 1 message broker + 2 bancos de dados**, todos orquestrados via Docker Compose em uma rede interna chamada `arch_network`.

### Fluxo completo

```
Usuário
  │
  ▼
[API Gateway :8000]
  │  POST /api/v1/analyses  (encaminha o arquivo)
  ▼
[Upload Service :8001]
  │  Salva arquivo em disco + registra no banco
  │  Publica mensagem na fila "diagram.process"
  ▼
[RabbitMQ :5672]
  │  Entrega mensagem assincronamente
  ▼
[Processing Service :8002]
  │  Consome a fila, lê o arquivo, envia ao Claude AI
  │  Recebe JSON estruturado da IA
  │  Chama Reports Service para salvar o relatório
  │  Atualiza status no Upload Service
  ▼
[Reports Service :8003]
  │  Persiste o relatório no banco
  ▼
[Usuário consulta GET /api/v1/reports/{id}]
```

---

## 1. API Gateway

**Pasta:** `api-gateway/`
**Porta:** `8000` (exposta para o host)
**Função:** Ponto de entrada único do sistema. Recebe requisições externas e as roteia para os serviços internos via HTTP.

### Pacotes

| Pacote | Versão | Para que serve |
|---|---|---|
| `fastapi` | 0.111.0 | Framework web — define os endpoints REST |
| `uvicorn[standard]` | 0.30.1 | Servidor ASGI que executa a aplicação FastAPI |
| `httpx` | 0.27.0 | Cliente HTTP assíncrono para chamar os outros serviços |
| `pydantic-settings` | 2.3.4 | Carrega variáveis de ambiente com tipagem |
| `python-multipart` | 0.0.9 | Suporte a upload de arquivos (`multipart/form-data`) |

### Endpoints

| Método | Rota | Ação |
|---|---|---|
| GET | `/health` | Verifica se o serviço está vivo |
| POST | `/api/v1/analyses` | Recebe arquivo (PDF/imagem) e repassa ao upload-service |
| GET | `/api/v1/analyses/{id}` | Consulta status de uma análise via upload-service |
| GET | `/api/v1/reports/{id}` | Busca relatório gerado via reports-service |

### Comunicação

- Recebe chamadas de: **usuário externo (navegador, Swagger, curl)**
- Chama: **upload-service** (`:8001`) e **reports-service** (`:8003`)
- Não possui banco de dados próprio
- Timeout das chamadas HTTP: **60 segundos**

### Dockerfile explicado

```dockerfile
FROM python:3.11-slim           # Imagem base enxuta do Python 3.11

WORKDIR /app                    # Define /app como diretório de trabalho

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
# Instala o curl — necessário para o healthcheck do Docker Compose
# (o compose verifica: curl -f http://localhost:8000/health)

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Instala dependências primeiro (antes do código) para aproveitar cache de camadas Docker

COPY . .                        # Copia todo o código do serviço

EXPOSE 8000                     # Documenta que o container escuta na porta 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# Inicia o servidor uvicorn escutando em todas as interfaces de rede do container
```

---

## 2. Upload Service

**Pasta:** `upload-service/`
**Porta:** `8001` (exposta para o host)
**Função:** Recebe arquivos do API Gateway, persiste no disco, registra no banco de dados e publica uma mensagem no RabbitMQ para iniciar o processamento assíncrono.

### Pacotes

| Pacote | Versão | Para que serve |
|---|---|---|
| `fastapi` | 0.111.0 | Framework web |
| `uvicorn[standard]` | 0.30.1 | Servidor ASGI |
| `sqlalchemy[asyncio]` | 2.0.31 | ORM assíncrono para acesso ao PostgreSQL |
| `asyncpg` | 0.29.0 | Driver PostgreSQL assíncrono (usado pelo SQLAlchemy) |
| `aio-pika` | 9.4.1 | Cliente RabbitMQ assíncrono para publicar mensagens |
| `pydantic-settings` | 2.3.4 | Variáveis de ambiente tipadas |
| `python-multipart` | 0.0.9 | Recepção de arquivos via formulário |
| `pytest` + `pytest-asyncio` | — | Testes unitários assíncronos |

### Endpoints

| Método | Rota | Ação |
|---|---|---|
| GET | `/health` | Healthcheck |
| POST | `/analyses` | Recebe arquivo, salva em disco, publica na fila |
| GET | `/analyses/{id}` | Retorna status da análise |
| PUT | `/analyses/{id}/status` | Atualiza status (chamado pelo processing-service) |

### Status possíveis de uma análise

```
Recebido → Em processamento → Analisado
                           ↘ Erro
```

### Comunicação

- Recebe chamadas de: **api-gateway** (`:8000`) e **processing-service** (`:8002`)
- Publica mensagens em: **RabbitMQ** fila `diagram.process`
- Banco de dados: **upload-db** (PostgreSQL `:5432`, banco `upload_db`)
- Volume compartilhado: `/uploads` (arquivos enviados ficam aqui, acessível também pelo processing-service)

### Dockerfile explicado

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
# curl para healthcheck do compose

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /uploads
# Cria o diretório onde os arquivos enviados serão armazenados no container

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
```

---

## 3. Processing Service

**Pasta:** `processing-service/`
**Porta:** `8002` (exposta para o host, mas raramente chamada diretamente)
**Função:** Serviço worker. Fica escutando a fila RabbitMQ, e ao receber uma mensagem, lê o arquivo do disco, envia ao Claude AI (Anthropic) e orquestra a criação do relatório.

### Pacotes

| Pacote | Versão | Para que serve |
|---|---|---|
| `fastapi` | 0.111.0 | Framework web (apenas para o endpoint `/health`) |
| `uvicorn[standard]` | 0.30.1 | Servidor ASGI |
| `aio-pika` | 9.4.1 | Consumidor RabbitMQ assíncrono |
| `anthropic` | 0.29.0 | SDK oficial da Anthropic para chamar o Claude AI |
| `httpx` | 0.27.0 | Cliente HTTP para chamar reports-service e upload-service |
| `pydantic-settings` | 2.3.4 | Variáveis de ambiente |
| `pytest` + `pytest-asyncio` | — | Testes |

### Lógica de processamento (use case)

Ao consumir uma mensagem da fila, o `ProcessDiagramUseCase` executa:

1. Atualiza status → **"Em processamento"** no upload-service
2. Lê o arquivo do volume `/uploads`
3. Converte o arquivo para base64
4. Envia ao Claude AI:
   - PDF → bloco do tipo `document`
   - Imagens → bloco do tipo `image`
5. Recebe JSON estruturado da IA com componentes, riscos e recomendações
6. Salva o relatório no reports-service via HTTP POST
7. Atualiza status → **"Analisado"** no upload-service
8. Em caso de erro: atualiza status → **"Erro"** com a mensagem da exceção

### Modelo de IA usado

**`claude-opus-4-6`** — configurado em `processing-service/app/config.py`

### Prompt enviado ao Claude

O sistema instrui o Claude a atuar como arquiteto de software especialista e retornar **somente JSON** com este formato:

```json
{
  "components": [{"name": "", "type": "", "description": ""}],
  "risks": [{"severity": "high|medium|low", "title": "", "description": ""}],
  "recommendations": [{"priority": "high|medium|low", "title": "", "description": ""}],
  "summary": "Visão geral em 2-3 frases"
}
```

### Comunicação

- Consome mensagens de: **RabbitMQ** fila `diagram.process`
- Lê arquivos de: volume compartilhado `/uploads`
- Chama: **reports-service** (`:8003`) para salvar relatório
- Chama: **upload-service** (`:8001`) para atualizar status
- Chama: **Anthropic API** (internet externa) com a `ANTHROPIC_API_KEY`

### Dockerfile explicado

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8002
# Este serviço não recebe chamadas externas diretas,
# mas expõe a porta para o healthcheck do Docker Compose

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
# O consumo da fila RabbitMQ é iniciado no "lifespan" do FastAPI,
# rodando como uma task assíncrona em paralelo com o servidor HTTP
```

---

## 4. Reports Service

**Pasta:** `reports-service/`
**Porta:** `8003` (exposta para o host)
**Função:** Responsável exclusivamente por persistir e recuperar relatórios gerados pela IA. É um serviço de armazenamento estruturado.

### Pacotes

| Pacote | Versão | Para que serve |
|---|---|---|
| `fastapi` | 0.111.0 | Framework web |
| `uvicorn[standard]` | 0.30.1 | Servidor ASGI |
| `sqlalchemy[asyncio]` | 2.0.31 | ORM assíncrono |
| `asyncpg` | 0.29.0 | Driver PostgreSQL assíncrono |
| `pydantic-settings` | 2.3.4 | Variáveis de ambiente |
| `pytest` + `pytest-asyncio` | — | Testes |

### Endpoints

| Método | Rota | Quem chama | Ação |
|---|---|---|---|
| GET | `/health` | Docker Compose | Healthcheck |
| POST | `/reports` | processing-service | Cria relatório (endpoint interno) |
| GET | `/reports/{analysis_id}` | api-gateway | Retorna relatório por ID da análise |

### Estrutura do relatório salvo

```json
{
  "id": "uuid",
  "analysis_id": "uuid",
  "components": [...],
  "risks": [...],
  "recommendations": [...],
  "summary": "texto",
  "created_at": "ISO 8601"
}
```

### Comunicação

- Recebe chamadas de: **processing-service** (criar) e **api-gateway** (consultar)
- Banco de dados: **reports-db** (PostgreSQL `:5432`, banco `reports_db`)
- Não publica em filas, não chama outros serviços

### Dockerfile explicado

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8003

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
```

---

## 5. RabbitMQ

**Imagem:** `rabbitmq:3.12-management-alpine`
**Portas:**
- `5672` — protocolo AMQP (comunicação entre serviços)
- `15672` — interface web de gerenciamento

**Credenciais:** usuário `guest` / senha `guest`

**Função:** Message broker que desacopla o upload do processamento. Quando o upload-service publica uma mensagem, o processing-service a consome de forma independente e assíncrona, sem que os dois precisem estar sincronizados.

**Fila usada:** `diagram.process` (durável — sobrevive a reinicializações)

**Interface web:** http://localhost:15672 — permite visualizar filas, mensagens pendentes, consumers ativos e throughput.

---

## 6. Bancos de Dados

### upload-db

| Item | Valor |
|---|---|
| Imagem | `postgres:15-alpine` |
| Banco | `upload_db` |
| Usuário | `postgres` / `postgres` |
| Usado por | upload-service |
| Armazena | registros de análises com status e metadados do arquivo |

### reports-db

| Item | Valor |
|---|---|
| Imagem | `postgres:15-alpine` |
| Banco | `reports_db` |
| Usuário | `postgres` / `postgres` |
| Usado por | reports-service |
| Armazena | relatórios completos com componentes, riscos e recomendações |

---

## 7. Volumes Docker

| Volume | Usado por | Conteúdo |
|---|---|---|
| `upload_db_data` | upload-db | Dados persistentes do PostgreSQL |
| `reports_db_data` | reports-db | Dados persistentes do PostgreSQL |
| `uploads_data` | upload-service + processing-service | Arquivos enviados pelos usuários (PDF/imagens) |

> O volume `uploads_data` é compartilhado entre upload-service e processing-service, permitindo que o processing leia o arquivo salvo pelo upload sem precisar transferi-lo via HTTP.

---

## 8. Resumo de Portas e Acessos

| Serviço | Porta | Interface | Acesso |
|---|---|---|---|
| API Gateway | 8000 | Swagger em `/docs` | http://localhost:8000/docs |
| Upload Service | 8001 | Swagger em `/docs` | http://localhost:8001/docs |
| Processing Service | 8002 | Swagger em `/docs` | http://localhost:8002/docs |
| Reports Service | 8003 | Swagger em `/docs` | http://localhost:8003/docs |
| RabbitMQ Management | 15672 | Interface gráfica web | http://localhost:15672 |
| RabbitMQ AMQP | 5672 | Protocolo interno | (não acessar pelo browser) |

> Para uso normal, acesse apenas **http://localhost:8000/docs** (API Gateway). Os demais são serviços internos expostos apenas para debug.
