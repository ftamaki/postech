# Pós Tech FIAP - Artificial Intelligence for Devs

Repositório central para os exercícios e desafios técnicos (Tech Challenges) desenvolvidos durante a pós-graduação.

---

## 📂 Projetos

### 1️⃣ Tech Challenge 1
*(Adicione aqui uma breve descrição do primeiro desafio)*

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
1.  **Agente Orquestrador (`src/agent_orchestrator.py`)**:
    - Decide dinamicamente entre consultar protocolos médicos ou dados de pacientes.
    - Mantém o contexto da conversa.
2.  **RAG (Retrieval-Augmented Generation)**:
    - Consulta documentos locais (`data/protocolo_medico_simulado.txt`) para responder perguntas técnicas com precisão.
3.  **Fine-Tuning (`src/fine_tunning.py`)**:
    - Script para ajuste fino de modelos (ex: `facebook/opt-125m`) usando dataset customizado (`data/fine_tuning_dataset.jsonl`).