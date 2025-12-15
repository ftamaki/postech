## 🚀 Como Executar

Para configurar e executar o projeto, siga os passos abaixo:

### Pré-requisitos

Certifique-se de ter o Python 3.9 ou superior instalado.

### 1. Clonar o Repositório

```bash
git clone https://github.com/flaviohbr/postech.git
cd postech/Exercicios/Tech\ Challenge\ 3
```

### 2. Configurar o Ambiente Virtual

É altamente recomendável usar um ambiente virtual para gerenciar as dependências.

```bash
python -m venv venv
source venv/bin/activate  # No Linux/macOS
# venv\Scripts\activate   # No Windows
```

### 3. Instalar Dependências

Instale todas as bibliotecas necessárias a partir do arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Configurar Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto (`postech/Exercicios/Tech Challenge 3/.env`) e adicione sua chave de API do Google Gemini:

```
GOOGLE_API_KEY="SUA_CHAVE_API_AQUI"
```

Substitua `"SUA_CHAVE_API_AQUI"` pela sua chave real.

### 5. Executar o Agente

Para iniciar o agente conversacional, execute o script principal:

```bash
python src/agent_orchestrator.py
```

O agente estará pronto para interagir no terminal.

### 6. Executar o Fine-Tuning (Opcional)

Se desejar realizar o fine-tuning de um modelo, execute o script `fine_tuning.py`. Certifique-se de que o dataset (`data/fine_tuning_dataset.jsonl`) e o modelo base (`facebook/opt-125m`) estejam acessíveis.

```bash
python src/fine_tuning.py
```

**Nota**: O fine-tuning pode exigir recursos computacionais significativos (GPU).

---

## 📁 Estrutura do Projeto
postech/Exercicios/Tech Challenge 3/
├── data/
│   ├── fine_tuning_dataset.jsonl         # Dataset para fine-tuning
│   └── protocolo_medico_simulado.txt     # Documento de protocolo médico para RAG
├── src/
│   ├── agent_orchestrator.py             # Script principal do agente conversacional
│   ├── fine_tuning.py                    # Script para fine-tuning de modelos
│   ├── chains.py                         # Definições de cadeias LangChain
│   ├── tools.py                          # Definições de ferramentas para o agente
│   └── utils.py                          # Funções utilitárias
├── .env.example                          # Exemplo de arquivo de variáveis de ambiente
├── requirements.txt                      # Dependências do projeto
└── README.md                             # Este arquivo
