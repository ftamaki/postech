from dotenv import load_dotenv                              # Carrega variáveis de ambiente de um arquivo .env
from pathlib import Path                                    # Manipulação de caminhos de arquivos
import sys

# LangChain Core - Componentes Básicos
from langchain_core.prompts import ChatPromptTemplate       # Criação de prompts para chat, com suporte a múltiplas mensagens
from langchain_core.runnables import RunnablePassthrough    # Passa a entrada diretamente para a saída sem modificação
from langchain_core.output_parsers import StrOutputParser   # Analisa a saída como uma string simples
from langchain_core.tools import tool                       # Decorador para definir ferramentas, ex: APIs ou funções utilitárias
from langchain_core.messages import HumanMessage, AIMessage # Representações de mensagens feiitas por humanos e IAs

# LangChain Components - LLMs, Embeddings, VectorStores
from langchain_community.vectorstores import FAISS          # FAISS é uma biblioteca para busca eficiente em grandes conjuntos de vetores - usada para recuperação de informações
from langchain_text_splitters import CharacterTextSplitter  # Divide textos em pedaços menores com base em caracteres
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings # Integração com modelos e embeddings do Google Generative AI

# LangGraph (nova API) - Orquestração de Fluxos de Trabalho
from langgraph.graph import StateGraph, END                 # Define grafos de estados para orquestração de fluxos de trabalho

# Local Tools - Importando a ferramenta de consulta ao banco de dados simulado
current_dir = Path(__file__).resolve().parent               # Diretório atual do arquivo (/home/ubuntu)
if str(current_dir) not in sys.path:                       # Adiciona o diretório atual ao sys.path
    sys.path.append(str(current_dir))                      # isso permite importar módulos de subdiretórios como 'src'

from db_simulado import consultar_paciente as consultar_paciente_db            # Ferramenta para consultar o banco de dados simulado de pacientes


# =============================================================
# 1. Configuração Inicial - LLM e Embeddings
# =============================================================
load_dotenv()                                              # Carrega variáveis de ambiente do arquivo .env

GEMINI_MODEL = "gemini-2.5-flash"                          # Modelo Gemini do Google
EMBEDDING_MODEL = "text-embedding-004"                     # Modelo de Embeddings do Google
print(f"\nUsando modelo Google Gemini: {GEMINI_MODEL}")

LLM = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.1)           # Instancia o LLM com o modelo especificado, temperatura baixa para respostas mais precisas
EMBEDDINGS = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)            # Instancia o gerador de embeddings com o modelo especificado
                                                                            # O gerador de embeddings converte texto em vetores numéricos para busca eficiente

# =============================================================
# 2. Configuração do RAG que quer dizer Retrieval-Augmented Generation
#    Em resumo, é uma técnica que combina recuperação de informações com geração de texto.
# =============================================================
def setup_rag_retriever(protocol_path: str = "data/protocolo_medico_simulado.txt"): # Configura o sistema RAG usando protocolos médicos simulados
    base_dir = Path(__file__).resolve().parent                                      # Diretório atual do arquivo
    absolute_path = base_dir.parent / protocol_path                                 # Caminho absoluto para o arquivo de protocolos médicos

    with open(absolute_path, "r", encoding="utf-8") as f:                           # Lê o conteúdo do arquivo de protocolos médicos
        protocolo = f.read()                                                        # Armazena o conteúdo do protocolo em uma variável

    text_splitter = CharacterTextSplitter(                                          # Configura o divisor de texto para dividir o protocolo em pedaços menores
        separator="\n\n",                                                           # separa por duplas quebras de linha - conforme o arquivo protocolo_medico_simulado.txt
        chunk_size=1000,                                                            # tamanho máximo de cada pedaço
        chunk_overlap=200,                                                          # sobreposição entre pedaços para manter contexto
        length_function=len,                                                        # função para medir o comprimento do texto
        is_separator_regex=False,                                                   # o separador não é uma expressão regular
    )

    texts = text_splitter.split_text(protocolo)                                     # Divide o protocolo em pedaços menores     

    vectorstore = FAISS.from_texts(texts, EMBEDDINGS)                               # Cria um índice FAISS a partir dos pedaços de texto e seus embeddings
    return vectorstore.as_retriever()


rag_retriever = setup_rag_retriever()                                               # Configura o recuperador RAG usando o protocolo médico simulado


# =============================================================
# 3. Criação da Tool RAG - Consulta ao Protocolo Médico Simulado
#   @tool é um decorador que transforma a função em uma ferramenta utilizável pelo agente.
#   Tool Rag é uma ferramenta que utiliza recuperação de informações para responder perguntas com base em um conjunto de dados específico.
# =============================================================
def create_rag_chain(retriever):                        # Cria uma cadeia RAG para responder perguntas usando o recuperador fornecido
    template = """
    Você é um assistente médico. Use o contexto fornecido (Protocolos Médicos)
    para responder à pergunta. Não invente informações.

    Contexto:
    {context}

    Pergunta: {question}

    Resposta:
    """ 
    # Prompt template para a cadeia RAG, orientando o LLM a usar o contexto recuperado para responder perguntas

    prompt = ChatPromptTemplate.from_template(template) # Cria o prompt de chat a partir do template definido, substituindo os placeholders {context} e {question}

    chain = (                                                       # chain serve para encadear múltiplos componentes juntos facilitando o fluxo de dados entre eles
        {"context": retriever, "question": RunnablePassthrough()}   # mapeia o contexto para o recuperador e a pergunta para a entrada direta
        | prompt                                                    # Gera o prompt formatado
        | LLM                                                       # Passa o prompt para o LLM para gerar a resposta
        | StrOutputParser()                                         # Analisa a saída do LLM como uma string simples                              
    )
    return chain                                                    # Retorna a cadeia RAG criada


rag_chain = create_rag_chain(rag_retriever)                         # Cria a cadeia RAG usando o recuperador configurado


@tool                                                               # Decorador que transforma a função em uma ferramenta utilizável pelo agente do RAG
def consultar_protocolo(pergunta: str) -> str:                      # consultar_protocolo é uma ferramenta que usa RAG para responder perguntas sobre protocolos médicos
    """Consulta o protocolo médico simulado para responder perguntas sobre condutas e procedimentos médicos.""" # Docstring explicativa da função
    return rag_chain.invoke(pergunta)                               # Invoca a cadeia RAG com a pergunta fornecida e retorna a resposta 

@tool
def consultar_paciente_tool(pergunta: str) -> str:
    """
    Consulta o banco de dados simulado de pacientes para obter informações como
    status de alerta, histórico médico e dados vitais.
    A pergunta deve ser formatada para que a ferramenta `consultar_paciente`
    possa extrair o ID do paciente (ex: P001).
    """
    # A função `consultar_paciente` (importada de src.db_simulado) é a implementação real.
    # Aqui, você pode adicionar lógica para extrair o ID do paciente da `pergunta`
    # antes de chamar a função real, mas por simplicidade, vamos passar a pergunta
    # diretamente para a função simulada, que deve ser capaz de extrair o ID.
    return consultar_paciente_db(pergunta)



# =============================================================
# 4. DEFINIÇÃO DO ESTADO DO AGENTE - estrutura de dados para armazenar o estado do agente
#   essa parte define como o agente mantém o contexto da conversa
# =============================================================
class AgentState(dict): # Define o estado do agente como um dicionário
    messages: list      # Lista de mensagens trocadas entre o humano e o agente


# =============================================================
# 5. NÓ PRINCIPAL DO AGENTE - lógica do agente para decidir quando e qual ferramenta usar
#   essa parte define o comportamento do agente
# =============================================================
TOOLS = { # Dicionário de ferramentas disponíveis para o agente
    "consultar_protocolo": consultar_protocolo, # Ferramenta para consultar protocolos médicos
    "consultar_paciente": consultar_paciente_tool,   # Ferramenta para consultar dados de pacientes
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
""" # Prompt do sistema definindo o comportamento e as regras do agente, incluindo as ferramentas disponíveis


def agente_node(state: AgentState) -> AgentState:   # Função que define o comportamento do agente em um nó do grafo, grafo que é usado para orquestrar o fluxo de trabalho
    last = state["messages"][-1]                    # Obtém a última mensagem da conversa

    # Se o último for mensagem do humano, pedir análise ao LLM
    if isinstance(last, HumanMessage):              # Verifica se a última mensagem foi enviada pelo usuário (humano)
        user_text = last.content                    # Extrai o conteúdo da mensagem enviada pelo usuário

        # Decide se deve usar ferramenta, e qual usar
        if "paciente" in user_text.lower() or "p00" in user_text.lower():   # Verifica se a mensagem contém termos relacionados a pacientes
            result = TOOLS["consultar_paciente"].invoke(user_text)          # Usa a ferramenta consultar_paciente para responder à pergunta
            ai_msg = AIMessage(content=result)                      
        elif "conduta" in user_text.lower() or "protocolo" in user_text.lower() or "protocolos" in user_text.lower():
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
