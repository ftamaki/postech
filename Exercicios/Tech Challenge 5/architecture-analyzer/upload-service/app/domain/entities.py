from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID


class AnalysisStatus(str, Enum):
    RECEBIDO = "Recebido"
    EM_PROCESSAMENTO = "Em processamento"
    ANALISADO = "Analisado"
    ERRO = "Erro"


@dataclass
class Analysis:
    id: UUID
    original_filename: str
    file_path: str
    file_type: str
    status: AnalysisStatus
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
