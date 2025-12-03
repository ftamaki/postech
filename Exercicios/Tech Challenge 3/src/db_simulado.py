from typing import List, Dict, Any
from langchain_core.tools import tool

# Simulação de um banco de dados de pacientes
# Em um ambiente real, esta seria uma conexão com um banco de dados SQL, NoSQL ou API.
PATIENT_DB: List[Dict[str, Any]] = [
    {
        "id": "P001",
        "nome": "João Silva",
        "idade": 65,
        "diagnostico_principal": "Sepse de foco pulmonar",
        "exames_recentes": {
            "lactato": 5.2, # Alto
            "hemocultura": "Positiva para Klebsiella pneumoniae",
            "pressao_arterial": "90/60 mmHg", # Hipotensão
            "frequencia_respiratoria": 25, # Alta
        },
        "historico_medico": "Diabetes Mellitus tipo 2, Hipertensão Arterial Sistêmica.",
        "status_alerta": "Emergência - Protocolo Sepse Ativado"
    },
    {
        "id": "P002",
        "nome": "Maria Oliveira",
        "idade": 48,
        "diagnostico_principal": "Crise Hipertensiva",
        "exames_recentes": {
            "lactato": 1.5, # Normal
            "hemocultura": "Negativa",
            "pressao_arterial": "195/110 mmHg", # Elevada
            "frequencia_respiratoria": 18, # Normal
        },
        "historico_medico": "Nega comorbidades. Não aderente ao tratamento.",
        "status_alerta": "Urgência - Necessita de ajuste de medicação oral"
    },
    {
        "id": "P003",
        "nome": "Carlos Souza",
        "idade": 72,
        "diagnostico_principal": "Dor Torácica",
        "exames_recentes": {
            "troponina": "0.01 ng/mL", # Normal
            "ecg": "Sem alterações isquêmicas agudas",
            "pressao_arterial": "130/80 mmHg",
            "frequencia_respiratoria": 16,
        },
        "historico_medico": "Doença arterial coronariana prévia. Angioplastia há 5 anos.",
        "status_alerta": "Investigação - Baixo risco para SCA"
    }
]

@tool
def consultar_paciente(paciente_id: str) -> str:
    """
    Consulta o banco de dados simulado de pacientes e retorna todas as informações
    relevantes para o ID do paciente fornecido.
    Útil para obter dados em tempo real sobre o estado clínico, exames e histórico do paciente.
    O 'paciente_id' deve ser uma string no formato 'PXXX', como 'P001'.
    """
    for paciente in PATIENT_DB:
        if paciente["id"] == paciente_id:
            # Retorna as informações formatadas como string para o LLM
            info = f"Informações do Paciente {paciente_id} - {paciente['nome']}:\n"
            info += f"Idade: {paciente['idade']}\n"
            info += f"Diagnóstico Principal: {paciente['diagnostico_principal']}\n"
            info += f"Status de Alerta: {paciente['status_alerta']}\n"
            info += f"Histórico Médico: {paciente['historico_medico']}\n"
            info += "Exames Recentes:\n"
            for chave, valor in paciente["exames_recentes"].items():
                info += f"  - {chave.replace('_', ' ').title()}: {valor}\n"
            return info
    
    return f"Paciente com ID {paciente_id} não encontrado no sistema."

# Exemplo de uso (apenas para teste local)
if __name__ == "__main__":
    # Para testar a função subjacente (a lógica de consulta) sem o LangChain Tool wrapper,
    # acessamos a função original através do atributo .func.
    print(consultar_paciente.func("P001"))
    print("-" * 20)
    print(consultar_paciente.func("P004"))
