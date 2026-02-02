import os
from pathlib import Path
from dotenv import load_dotenv

# Ponto de partida: este arquivo = .../Tech Challenge 4/src/azure/config.py
# Queremos: .../postech/.env  (ou seja: subir 4 níveis)
ROOT = Path(__file__).resolve().parents[4]  # .../postech
ENV_PATH = ROOT / ".env"

# Carrega explicitamente o .env do caminho informado
load_dotenv(dotenv_path=ENV_PATH)

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "techchallenge4")
