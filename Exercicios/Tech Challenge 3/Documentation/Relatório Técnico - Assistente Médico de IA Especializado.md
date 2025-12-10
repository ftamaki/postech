# Relatório Técnico: Assistente Médico de IA Especializado

## 1. Introdução

Este relatório descreve a arquitetura e a implementação do Assistente Médico de IA desenvolvido para o Tech Challenge 3. O objetivo principal foi criar um agente inteligente capaz de fornecer informações médicas precisas, aderentes a protocolos e contextualizadas com dados de pacientes, demonstrando proficiência em orquestração de LLMs, fine-tuning e implementação de requisitos de segurança.

## 2. Arquitetura do Agente (LangGraph)

O agente utiliza o **LangGraph** para definir um fluxo de trabalho robusto e modular. O fluxo principal é composto por um único nó de decisão (`agente_node`) que executa as seguintes etapas sequenciais:

1.  **Guardrail (Limites de Atuação):** Verifica se a pergunta do usuário está dentro do escopo médico.
2.  **Roteamento:** Decide qual ferramenta ou ação executar com base na intenção da pergunta.
3.  **Execução da Ferramenta:** Chama a ferramenta apropriada (`consultar_paciente` ou `consultar_protocolo`).
4.  **Geração de Resposta:** Utiliza o LLM Fine-Tuned para formatar a resposta final.

## 3. Implementação dos Requisitos Técnicos

### 3.1. Fine-Tuning do LLM (Requisito 1)

*   **Modelo Base:** `facebook/opt-125m` (substituível por modelos mais robustos como LLaMA-3).
*   **Técnica:** **LoRA (Low-Rank Adaptation)**, implementada via biblioteca `peft` e `trl` (SFTTrainer).
*   **Dataset:** Um dataset sintético no formato JSONL foi criado a partir de protocolos médicos simulados, utilizando o formato de *Instruction-Tuning* para especializar o modelo na linguagem e nos protocolos clínicos.
*   **Resultado:** O modelo fine-tuned é carregado no `agent_orchestrator.py` e usado como o LLM principal.

### 3.2. Segurança e Validação (Requisitos 2 e 3)

#### A. Limites de Atuação (Guardrail)

*   **Mecanismo:** Uma função auxiliar (`check_medical_scope`) verifica a presença de palavras-chave médicas e a ausência de palavras-chave não médicas na entrada do usuário.
*   **Ação:** Se a pergunta for considerada fora do escopo, o agente retorna uma mensagem de recusa padronizada, garantindo que o sistema não seja usado para fins não médicos.

#### B. Logging

*   **Mecanismo:** O módulo `logging` do Python foi configurado para registrar eventos no console e no arquivo `agent_log.log`.
*   **Informações Registradas:** Entrada do usuário, decisão de roteamento (qual ferramenta foi chamada), e o tamanho da resposta final.

#### C. Explainability

*   **Mecanismo:** Implementado na lógica de roteamento para perguntas de paciente.
*   **Processo:** O resultado da ferramenta `consultar_paciente` é injetado no *prompt* do LLM Fine-Tuned, forçando o modelo a basear sua resposta nos dados factuais do paciente.

## 4. Próximos Passos e Aprimoramentos

1.  **Explainability do RAG:** Aprimorar a ferramenta `consultar_protocolo` para que a resposta final do agente inclua a citação exata do *chunk* de texto do protocolo que foi usado, atendendo plenamente ao requisito de explainability.
2.  **Melhoria do Roteamento:** Substituir a lógica de roteamento baseada em palavras-chave por um **LLM Router** (um pequeno LLM treinado para classificar a intenção da pergunta), o que tornaria o agente mais robusto.
3.  **Avaliação de Desempenho:** Incluir métricas de avaliação (ex: ROUGE, BLEU) para medir a melhoria do LLM após o fine-tuning.

## 5. Conclusão

O Assistente Médico de IA demonstra a aplicação prática de técnicas de IA Generativa para resolver problemas complexos no domínio da saúde. A combinação de fine-tuning, orquestração e mecanismos de segurança/validação estabelece uma base sólida para um sistema de suporte à decisão clínica.
