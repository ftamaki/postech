import os
from dotenv import load_dotenv
from pathlib import Path

# LangChain utilities
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Google Gemini Integration
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# =============================================================
# 1. Carregar variáveis de ambiente
# =============================================================
load_dotenv()
# NOTA: Para o ChatGoogleGenerativeAI funcionar, a variável de ambiente GOOGLE_API_KEY deve estar configurada.

# =============================================================
# 2. Configuração do LLM e Embeddings
# =============================================================

# Seleção do modelo Gemini
GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "text-embedding-004" # Modelo de embeddings do Google
print(f"\nUsando modelo Google Gemini: {GEMINI_MODEL}")

# Criar instância do modelo Gemini
LLM = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.1)

# Usamos GoogleGenerativeAIEmbeddings
EMBEDDINGS = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# =============================================================
# 3. Criar VectorStore FAISS
# =============================================================
def setup_vector_store(protocol_path: str = "data/protocolo_medico_simulado.txt"):
    """
    Carrega o protocolo médico, divide em chunks e cria o Vector Store FAISS.
    """
    # Usa pathlib para construir o caminho de forma robusta e compatível com diferentes OS.
    # O caminho é construído a partir do diretório do script atual.
    base_dir = Path(__file__).resolve().parent
    # Navega para a raiz do projeto (dois níveis acima) e depois para o arquivo de dados
    absolute_path = base_dir.parent.parent / protocol_path

    with open(absolute_path, "r", encoding="utf-8") as f:
        protocolo = f.read()

    # Splitter
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_text(protocolo)

    # FAISS VectorStore
    vectorstore = FAISS.from_texts(texts, EMBEDDINGS)
    return vectorstore

# =============================================================
# 4. Criar pipeline RAG usando LCEL
# =============================================================
def create_rag_chain(vectorstore: FAISS):
    """
    Cria a chain RAG usando LangChain Expression Language (LCEL).
    """
    template = """
    Você é um assistente médico de um hospital. Sua função é auxiliar médicos em condutas clínicas.
    Use o contexto fornecido abaixo (Protocolos Médicos) para responder à pergunta.
    Se a resposta não puder ser encontrada no contexto, diga que não tem informações suficientes, 
    mas NUNCA invente uma resposta.
    
    Contexto (Protocolos Médicos):
    {context}
    
    Pergunta: {question}
    
    Resposta:
    """

    prompt = PromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | LLM
        | StrOutputParser()
    )

    return chain

# =============================================================
# 5. Execução principal
# =============================================================
if __name__ == "__main__":
    vectorstore = setup_vector_store()
    rag_chain = create_rag_chain(vectorstore)

    # --------------------------
    # Teste 1
    q1 = "Qual a conduta para Urgência Hipertensiva e qual o objetivo de redução da PA?"
    print(f"\n[PERGUNTA 1]: {q1}")
    print("\n[RESPOSTA 1]:\n", rag_chain.invoke(q1))

    # --------------------------
    # Teste 2
    q2 = "Quais exames iniciais devem ser solicitados para um paciente com suspeita de Sepse e qual a conduta de tratamento em 1 hora?"
    print(f"\n[PERGUNTA 2]: {q2}")
    print("\n[RESPOSTA 2]:\n", rag_chain.invoke(q2))

    # --------------------------
    # Teste 3
    q3 = "Posso prescrever Nitroprussiato de Sódio para um paciente no ambulatório?"
    print(f"\n[PERGUNTA 3]: {q3}")
    print("\n[RESPOSTA 3]:\n", rag_chain.invoke(q3))

    # --------------------------
    # Teste 4
    q4 = "Qual o melhor tratamento para dor de cabeça?"
    print(f"\n[PERGUNTA 4]: {q4}")
    print("\n[RESPOSTA 4]:\n", rag_chain.invoke(q4))
