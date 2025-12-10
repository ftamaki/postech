import os
from dotenv import load_dotenv
from pathlib import Path
import sys
import logging # Novo para logging
import torch

# Hugging Face/Deep Learning Imports
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel

# LangChain Core
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# LangChain Components
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Mantido para Embeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline # Novo para o modelo local

# LangGraph
from langgraph.graph import StateGraph, END

# Adiciona o diretório raiz do projeto ao sys.path para que as importações locais funcionem
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Local Tools
from src.db_simulado import consultar_paciente

# =============================================================
# 0. Configuração de Logging
# =============================================================
LOG_FILE = Path(__file__).resolve().parent.parent / "agent_log.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info("--- Novo Ciclo de Execução do Agente ---")
# =============================================================


# =============================================================
# 1. Configuração Inicial
# =============================================================
load_dotenv()

# --- Configuração do Modelo Fine-Tuned ---
# Caminhos
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINE_TUNED_MODEL_PATH = PROJECT_ROOT / "fine_tuned_model" / "med-assistant-lora"
BASE_MODEL_NAME = "facebook/opt-125m" # Deve ser o mesmo usado no fine_tuning.py

print(f"\nCarregando modelo Fine-Tuned de: {FINE_TUNED_MODEL_PATH}")

# 1. Configuração de Quantização (Deve ser a mesma usada no fine-tuning)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# 2. Carregar o Modelo Base e o Tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 3. Carregar os Adaptadores LoRA
model = PeftModel.from_pretrained(base_model, FINE_TUNED_MODEL_PATH)
model = model.merge_and_unload() # Opcional: fundir os pesos LoRA no modelo base

# 4. Criar o Pipeline e o LLM do LangChain
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    torch_dtype=torch.float16,
    device_map="auto",
)

# O LLM agora é o modelo Fine-Tuned
LLM = HuggingFacePipeline(pipeline=pipe)
# -----------------------------------------

# Configuração de Embeddings (Mantido o Gemini para consistência com o RAG existente)
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDINGS = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


# =============================================================
# 2. Configuração do RAG
# =============================================================
def setup_rag_retriever(protocol_path: str = "data/protocolo_medico_simulado.txt"):
    base_dir = Path(__file__).resolve().parent
    absolute_path = base_dir.parent / protocol_path

    with open(absolute_path, "r", encoding="utf-8") as f:
        protocolo = f.read()

    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_text(protocolo)

    vectorstore = FAISS.from_texts(texts, EMBEDDINGS)
    return vectorstore.as_retriever()


rag_retriever = setup_rag_retriever()


# =============================================================
# 3. Criação da Tool RAG
# =============================================================
def create_rag_chain(retriever):
    # O prompt foi ajustado para ser compatível com o formato de resposta do LLM Fine-Tuned
    template = """
    Você é um assistente médico. Use o contexto fornecido (Protocolos Médicos)
    para responder à pergunta. Não invente informações.

    Contexto:
    {context}

    Pergunta: {question}

    Resposta:
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | LLM
        | StrOutputParser()
    )
    return chain


rag_chain = create_rag_chain(rag_retriever)


@tool
def consultar_protocolo(pergunta: str) -> str:
    """
    Consulta a base de conhecimento de Protocolos Médicos (RAG) para obter informações
    sobre condutas clínicas, diagnósticos e tratamentos padronizados.
    Use esta ferramenta quando a pergunta for sobre um protocolo médico geral,
    e não sobre um paciente específico.
    """
    return rag_chain.invoke(pergunta)


# =============================================================
# 4. DEFINIÇÃO DO ESTADO DO AGENTE
# =============================================================
class AgentState(dict):
    messages: list


# =============================================================
# 5. NÓ PRINCIPAL DO AGENTE
# =============================================================
TOOLS = {
    "consultar_protocolo": consultar_protocolo,
    "consultar_paciente": consultar_paciente,
}

# O SYSTEM_PROMPT é mantido, mas o LLM Fine-Tuned agora o interpreta
SYSTEM_PROMPT = """
Você é um Assistente Médico de IA altamente especializado.
Ferramentas disponíveis:
- consultar_protocolo
- consultar_paciente

Regras:
- Se a pergunta for sobre um paciente, use consultar_paciente.
- Se for sobre protocolos médicos, use consultar_protocolo.
- Se nenhuma ferramenta for necessária, responda diretamente.
"""


# Função auxiliar para verificar se a pergunta está dentro do escopo médico
def check_medical_scope(user_text: str) -> bool:
    # Palavras-chave médicas comuns
    medical_keywords = ["saúde", "médico", "doença", "diagnóstico", "tratamento", "sintoma", "protocolo", "paciente", "conduta", "exame", "medicação", "clínico"]
    
    # Palavras-chave não médicas (para guardrail)
    non_medical_keywords = ["futebol", "receita", "política", "notícias", "filme", "música", "código", "programação"]
    
    text_lower = user_text.lower()
    
    # Verifica se há palavras-chave médicas
    is_medical = any(keyword in text_lower for keyword in medical_keywords)
    
    # Verifica se há palavras-chave não médicas
    is_non_medical = any(keyword in text_lower for keyword in non_medical_keywords)
    
    # Regra simples: deve conter algo médico e não deve ser dominado por algo não médico
    return is_medical and not is_non_medical

def agente_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    
    # Log: Início do nó
    logger.info(f"Nó 'agente' iniciado. Última mensagem: {last.__class__.__name__}")

    # Se o último for mensagem do humano, pedir análise ao LLM
    if isinstance(last, HumanMessage):
        user_text = last.content
        logger.info(f"Entrada do Usuário: {user_text}")

        # --- GUARDRAIL: Limites de Atuação ---
        if not check_medical_scope(user_text):
            guardrail_response = "Desculpe, sou um Assistente Médico de IA e só posso responder a perguntas relacionadas à saúde, protocolos médicos e informações de pacientes. Por favor, reformule sua pergunta."
            logger.warning(f"GUARDRAIL ATIVADO: Pergunta fora do escopo médico. Resposta: {guardrail_response}")
            ai_msg = AIMessage(content=guardrail_response)
            
        else:
            # --- Lógica de Roteamento Aprimorada ---
            # 1. Tenta extrair o ID do paciente (P00X)
            import re
            match = re.search(r'(p\d{3})', user_text.lower())
            
            if match:
                patient_id = match.group(1).upper()
                logger.info(f"Decisão: Chamando ferramenta 'consultar_paciente' com ID: {patient_id}")
                # Acessa a função subjacente da Tool para evitar o erro de 'StructuredTool' object is not callable
                result = TOOLS["consultar_paciente"].func(patient_id)
                
                # Adiciona a informação do paciente ao prompt para o LLM
                if "não encontrado" not in result:
                    # Requisito de Explainability: LLM formata a resposta com base nos dados do paciente
                    full_prompt = f"{SYSTEM_PROMPT}\n\nInformação do Paciente: {result}\n\nPergunta: {user_text}\n\nResposta:"
                    final_response = LLM.invoke(full_prompt)
                    ai_msg = AIMessage(content=final_response)
                else:
                    ai_msg = AIMessage(content=result) # Retorna a mensagem de erro do DB
                    
            elif "conduta" in user_text.lower() or "protocolo" in user_text.lower():
                logger.info("Decisão: Chamando ferramenta 'consultar_protocolo' (RAG)")
                # Acessa a função subjacente da Tool para evitar o erro de 'StructuredTool' object is not callable
                result = TOOLS["consultar_protocolo"].func(user_text)
                ai_msg = AIMessage(content=result)
            else:
                logger.info("Decisão: Resposta direta do LLM Fine-Tuned")
                # resposta direta sem ferramenta
                # O LLM Fine-Tuned é invocado diretamente
                full_prompt = f"{SYSTEM_PROMPT}\n\nPergunta: {user_text}\n\nResposta:"
                result = LLM.invoke(full_prompt)
                ai_msg = AIMessage(content=result)
        
        # Log: Saída do nó
        logger.info(f"Saída do Nó: Resposta gerada (Tamanho: {len(ai_msg.content)} caracteres)")

        return {"messages": state["messages"] + [ai_msg]}

    return state


# =============================================================
# 6. MONTAGEM DO GRAFO
# =============================================================
workflow = StateGraph(AgentState)
workflow.add_node("agente", agente_node)
workflow.set_entry_point("agente")
workflow.add_edge("agente", END)

agent_app = workflow.compile()


# =============================================================
# 7. Função principal de execução
# =============================================================
def run_agent(user_input: str):
    initial = {"messages": [HumanMessage(content=user_input)]}
    result = agent_app.invoke(initial)
    return result["messages"][-1].content


# =============================================================
# 8. Testes
# =============================================================
if __name__ == "__main__":
    print("\n===== TESTE 1: PROTOCOLO =====")
    q1 = "Qual a conduta para Urgência Hipertensiva e qual o objetivo de redução da PA?"
    print(run_agent(q1))

    print("\n===== TESTE 2: PACIENTE =====")
    q2 = "Qual o status de alerta e o histórico médico do paciente P001?"
    print(run_agent(q2))

    print("\n===== TESTE 3: MISTO =====")
    q3 = "O paciente P002 está com pressão 195/110. Qual a conduta recomendada?"
    print(run_agent(q3))
