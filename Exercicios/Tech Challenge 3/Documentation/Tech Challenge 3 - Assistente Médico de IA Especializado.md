# Tech Challenge 3: Assistente Médico de IA Especializado

## 1. Visão Geral do Projeto

Este projeto implementa um Assistente Médico de IA especializado, capaz de responder a consultas sobre protocolos clínicos e fornecer informações contextualizadas sobre pacientes, utilizando técnicas avançadas de Inteligência Artificial Generativa.

O assistente é construído sobre o framework **LangChain/LangGraph** e utiliza um **Modelo de Linguagem Grande (LLM) Fine-Tuned** para garantir respostas precisas e aderentes aos protocolos internos.

## 2. Requisitos Implementados

| Requisito | Status | Implementação |
| :--- | :--- | :--- |
| **Fine-Tuning do LLM** | **CONCLUÍDO** | O modelo base (`facebook/opt-125m`) foi ajustado com a técnica **LoRA** (Low-Rank Adaptation) em um dataset de protocolos médicos simulados. |
| **Orquestração (LangGraph)** | **CONCLUÍDO** | O fluxo de trabalho é orquestrado pelo LangGraph, que roteia a pergunta do usuário para a ferramenta apropriada (RAG, Consulta de Paciente ou Resposta Direta). |
| **Limites de Atuação (Guardrail)** | **CONCLUÍDO** | Implementado um guardrail que restringe o agente a responder apenas perguntas relacionadas à saúde e medicina, recusando tópicos fora do escopo. |
| **Explainability** | **PARCIAL** | O agente utiliza o resultado da ferramenta `consultar_paciente` como contexto para o LLM, garantindo que a resposta seja baseada em dados factuais. (Ainda pode ser aprimorado para o RAG). |
| **Logging** | **CONCLUÍDO** | Um sistema de logging registra as entradas, decisões de roteamento e saídas do agente no arquivo `agent_log.log`. |

## 3. Estrutura do Projeto

```
.
├── data/
│   ├── fine_tuning_dataset.jsonl  # Dataset gerado para o fine-tuning
│   └── protocolo_medico_simulado.txt
├── fine_tuned_model/              # Diretório onde o modelo LoRA é salvo
│   └── med-assistant-lora/
├── src/
│   ├── agent_orchestrator.py      # O coração do agente (LangGraph, RAG, Tools, Guardrail)
│   ├── db_simulado.py             # Simulação do banco de dados de pacientes (Tools)
│   └── __init__.py
├── agent_log.log                  # Arquivo de log gerado
├── fine_tuning.py                 # Script para realizar o fine-tuning do LLM (LoRA)
├── prepare_dataset.py             # Script para preparar o dataset de fine-tuning
└── requirements.txt
```

## 4. Instalação e Execução

### 4.1. Configuração do Ambiente

Recomendamos o uso de um ambiente Conda com GPU NVIDIA para o fine-tuning.

```bash
# 1. Crie e ative o ambiente Conda
conda create -n techchallenge3 python=3.10 -y
conda activate techchallenge3

# 2. Instale as dependências (incluindo PyTorch com CUDA)
# Use o comando pip para o PyTorch com a versão do seu CUDA (ex: cu118)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Instale as demais dependências
pip install -r requirements.txt
```

### 4.2. Fine-Tuning do Modelo

1.  **Gere o Dataset:**
    ```bash
    python prepare_dataset.py
    ```
2.  **Execute o Fine-Tuning (Requer GPU):**
    ```bash
    python fine_tuning.py
    ```
    O modelo será salvo em `fine_tuned_model/med-assistant-lora`.

### 4.3. Execução do Agente

Execute o orquestrador a partir do diretório `src/`:

```bash
cd src
python agent_orchestrator.py
```

A saída mostrará os testes de execução e o arquivo `agent_log.log` será atualizado.

## 5. Próximos Passos (Sugestões de Aprimoramento)

1.  **Explainability do RAG:** Modificar a `consultar_protocolo` para retornar a fonte (o chunk de texto) e usar o LLM para formatar a resposta, citando a fonte.
2.  **Melhoria do Guardrail:** Usar um LLM (ou um classificador) para determinar a intenção da pergunta com maior precisão, em vez de apenas palavras-chave.
3.  **Relatório Técnico:** Elaborar o relatório detalhado (Requisito de Entrega).
4.  **Vídeo de Demonstração:** Gravar a demonstração do fluxo completo.
