# Hackathon FIAP PosTech — IADT + SOAT

## Visão Geral

Construir um **sistema back-end** capaz de receber diagramas de arquitetura (imagem ou PDF), processá-los com IA e gerar um relatório técnico estruturado com componentes identificados, riscos arquiteturais e recomendações.

---

## Fluxo Principal (MVP)

```
[Upload do Diagrama] → [Processamento] → [Análise com IA] → [Geração de Relatório] → [Consulta de Status]
```

---

## Requisitos Funcionais

| # | Funcionalidade | Detalhes |
|---|---------------|---------|
| 1 | Upload de diagrama | Aceitar imagem ou PDF |
| 2 | Criação de processo de análise | Iniciar pipeline ao receber o arquivo |
| 3 | Consulta de status | `Recebido` → `Em processamento` → `Analisado` / `Erro` |
| 4 | Relatório técnico | Componentes identificados, riscos arquiteturais, recomendações |

---

## Requisitos Técnicos

### Arquitetura de Software (SOAT)

- [ ] Arquitetura baseada em **microsserviços**
- [ ] Comunicação via **REST**
- [ ] Ao menos **um fluxo assíncrono** (fila ou mensageria)
- [ ] Padrão: **Clean Architecture** ou **Arquitetura Hexagonal**
- [ ] Cada serviço deve ter:
  - [ ] Responsabilidade clara
  - [ ] Banco de dados próprio
  - [ ] Testes automatizados

#### Serviços Mínimos Sugeridos

| Serviço | Responsabilidade |
|---------|-----------------|
| API Gateway / BFF | Ponto de entrada, roteamento |
| Upload e Orquestração | Receber arquivo, enfileirar processamento |
| Processamento | Extrair informações do diagrama, acionar IA |
| Relatórios | Persistir e expor relatório gerado |

---

### Inteligência Artificial (IADT)

Implementar **ao menos uma** das abordagens:

- [ ] Detecção de componentes arquiteturais em imagens
- [ ] Classificação de riscos a partir de regras + ML
- [ ] **LLM para geração de relatório** com guardrails (controle de entrada/saída, mitigação de alucinações)
- [ ] Análise textual via **prompt engineering** com validação de prompts, restrições de formato e avaliação de consistência

#### Requisitos Mínimos de IA

- [ ] Pipeline claro de IA documentado
- [ ] Justificativa da abordagem escolhida
- [ ] Demonstração prática da análise
- [ ] Discussão das limitações do modelo

---

### Integração IA + Sistema

Deixar claro no código e documentação:

- [ ] Como a IA é acionada no fluxo
- [ ] Como o sistema trata falhas da IA
- [ ] Como o resultado da IA é persistido
- [ ] Como o relatório é gerado a partir da análise

---

### Infraestrutura e DevOps

- [ ] Docker (Dockerfile por serviço)
- [ ] Docker Compose **ou** Kubernetes
- [ ] Pipeline CI/CD com:
  - [ ] Build
  - [ ] Testes
  - [ ] Deploy (local ou cloud)

---

### Qualidade e Observabilidade

- [ ] Logs estruturados
- [ ] Tratamento de erros
- [ ] Testes unitários
- [ ] README explicativo

---

## Entregáveis

### 1. Código (Repositório Git)

- [ ] Código-fonte de todos os serviços
- [ ] Dockerfile por serviço
- [ ] `docker-compose.yml` ou manifestos Kubernetes
- [ ] Pipeline de CI/CD (ex: GitHub Actions)

### 2. Documentação (README)

- [ ] Descrição do problema
- [ ] Arquitetura proposta
- [ ] Fluxo da solução
- [ ] Instruções de execução
- [ ] Diagrama de arquitetura
- [ ] **Seção Segurança** (obrigatória):
  - [ ] Requisitos básicos de segurança adotados
  - [ ] Validação e tratamento de entradas não confiáveis
  - [ ] Uso controlado da IA (escopo e previsibilidade das respostas)
  - [ ] Tratamento seguro de falhas ou comportamentos inesperados da IA
  - [ ] Práticas de segurança na comunicação entre serviços
  - [ ] Riscos e limitações de segurança identificados

### 3. Vídeo (até 15 minutos)

- [ ] Apresentação da arquitetura
- [ ] Demonstração do fluxo completo
- [ ] Upload de um diagrama de exemplo
- [ ] Processamento e análise em tempo real (ou gravado)
- [ ] Exibição do relatório gerado

---

## Divisão de Responsabilidades

| Área | Time | Tarefas |
|------|------|---------|
| **Arquitetura** | SOAT | Design dos microsserviços, comunicação, persistência |
| **Infra / DevOps** | SOAT | Docker, CI/CD, observabilidade |
| **IA / Processamento** | IADT | Pipeline de IA, extração de informações, relatório |
| **Avaliação da IA** | IADT | Guardrails, limitações, avaliação de consistência |

---

## Sugestão de Stack

> Adaptar conforme o time decidir.

- **API Gateway**: FastAPI ou Node.js (Express)
- **Fila assíncrona**: RabbitMQ ou AWS SQS ou Redis Streams
- **IA**: Claude API (Anthropic) ou OpenAI Vision — multimodal para análise de imagem/PDF
- **Banco de dados**: PostgreSQL (por serviço) ou MongoDB
- **Infraestrutura**: Docker Compose (local) + GitHub Actions (CI/CD)

---

## Checklist Final

- [ ] Todos os serviços sobem com `docker-compose up`
- [ ] É possível fazer upload de um diagrama via API
- [ ] Status do processamento é consultável
- [ ] Relatório é retornado com componentes, riscos e recomendações
- [ ] CI/CD rodando (build + testes)
- [ ] README completo com seção de Segurança
- [ ] Vídeo gravado (até 15 min)
