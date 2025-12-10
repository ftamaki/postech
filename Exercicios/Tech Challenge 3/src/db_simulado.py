from langchain_core.tools import tool
import json

# =============================================================
# BANCO DE DADOS SIMULADO (MOCK)
# =============================================================

DB_PACIENTES = {
    "P001": {
        "nome": "João Silva",
        "idade": 45,
        "historico": ["Hipertensão", "Diabetes Tipo 2"],
        "medicamentos": ["Losartana 50mg", "Metformina 850mg"],
        "alergias": ["Penicilina"],
        "status_alerta": "BAIXO"
    },
    "P002": {
        "nome": "Maria Oliveira",
        "idade": 62,
        "historico": ["Insuficiência Cardíaca", "Asma"],
        "medicamentos": ["Furosemida 40mg", "Salbutamol"],
        "alergias": ["Nenhuma"],
        "status_alerta": "MÉDIO"
    },
    "P003": {
        "nome": "Carlos Souza",
        "idade": 33,
        "historico": ["Nenhum registro crônico"],
        "medicamentos": [],
        "alergias": ["Dipirona"],
        "status_alerta": "NENHUM"
    }
}

@tool
def consultar_paciente(id_paciente: str) -> str:
    """
    Consulta o banco de dados de pacientes pelo ID (ex: P001, P002).
    Retorna os dados clínicos, histórico e status do paciente.
    """
    # Normaliza o ID para maiúsculo e remove espaços
    id_clean = id_paciente.strip().upper()
    
    # Tenta encontrar o ID direto ou dentro da string fornecida
    import re
    match = re.search(r'(P\d{3})', id_clean)
    if match:
        id_clean = match.group(1)

    paciente = DB_PACIENTES.get(id_clean)

    if paciente:
        return json.dumps(paciente, indent=2, ensure_ascii=False)
    else:
        return f"Paciente com ID '{id_clean}' não encontrado no banco de dados."