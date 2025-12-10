import os
from dotenv import load_dotenv
from pathlib import Path
import sys

# LangChain Core
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# LangChain Components
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# LangGraph (nova API)
from langgraph.graph import StateGraph, END

# Local Tools
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.db_simulado import consultar_paciente


# =============================================================
# 1. Configuração Inicial
# =============================================================
load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "text-embedding-004"
print(f"\nUsando modelo Google Gemini: {GEMINI_MODEL}")

LLM = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.1)
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
    """Consulta o protocolo médico simulado para responder perguntas sobre condutas e procedimentos médicos."""
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


def agente_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]

    # Se o último for mensagem do humano, pedir análise ao LLM
    if isinstance(last, HumanMessage):
        user_text = last.content

        # Decide se deve usar ferramenta
        if "paciente" in user_text.lower() or "p00" in user_text.lower():
            result = TOOLS["consultar_protocolo"].invoke(user_text)
            ai_msg = AIMessage(content=result)
        elif "conduta" in user_text.lower() or "protocolo" in user_text.lower():
            result = TOOLS["consultar_protocolo"].invoke(user_text)
            ai_msg = AIMessage(content=result)
        else:
            # resposta direta sem ferramenta
            result = LLM.invoke([HumanMessage(content=user_text)])
            ai_msg = result

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
