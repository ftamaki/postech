# Simulação de um banco de dados de pacientes para consultas.
# Este módulo fornece funções para simular a consulta de dados de pacientes com base em perguntas.
# Cada paciente é identificado por um ID único (P001, P002, etc.).
# As respostas são pré-definidas para fins de simulação.
import re

def consultar_paciente(pergunta: str) -> str:
    """
    Simula a consulta a um banco de dados de pacientes.
    Extrai o ID do paciente da pergunta e retorna dados simulados.
    """
    # Tenta encontrar um ID de paciente no formato P00X
    match = re.search(r"(P\d{3})", pergunta, re.IGNORECASE)
    
    if match:
        paciente_id = match.group(1).upper()
        
        if paciente_id == "P001": # Exemplo de paciente com sepse
            return f"""
            Dados do Paciente {paciente_id}:
            - Status de Alerta: ALERTA VERMELHO (Sepse)
            - Histórico Médico: Diabetes Mellitus Tipo 2, Hipertensão Arterial Sistêmica.
            - Últimos Sinais Vitais: PA 90/60 mmHg, FC 110 bpm, FR 24 irpm, Temp 38.5°C.
            """
        elif paciente_id == "P002": # Exemplo de paciente com risco de hipertensão
            return f"""
            Dados do Paciente {paciente_id}:
            - Status de Alerta: ALERTA AMARELO (Risco de Hipertensão)
            - Histórico Médico: Obesidade, Sedentarismo.
            - Últimos Sinais Vitais: PA 195/110 mmHg, FC 80 bpm, FR 16 irpm, Temp 36.5°C.
            """
        else:
            return f"Paciente {paciente_id} não encontrado no banco de dados simulado." # Paciente não existe
    else:
        return "Não foi possível identificar o ID do paciente na pergunta." # ID não encontrado na pergunta
