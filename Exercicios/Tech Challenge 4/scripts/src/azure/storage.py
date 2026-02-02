# -*- coding: utf-8 -*-
from pathlib import Path
from azure.storage.blob import BlobServiceClient

try:
    from .config import AZURE_STORAGE_CONNECTION_STRING, AZURE_CONTAINER_NAME
except ImportError:  # running as a script
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from config import AZURE_STORAGE_CONNECTION_STRING, AZURE_CONTAINER_NAME

def _client():
    if not AZURE_STORAGE_CONNECTION_STRING:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING não definido.")
    return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

def ensure_container(container_name: str = AZURE_CONTAINER_NAME):
    svc = _client()
    c = svc.get_container_client(container_name)
    if not c.exists():
        c.create_container()
    return c

def upload_file(local_path: str, blob_name: str, container_name: str = AZURE_CONTAINER_NAME):
    c = ensure_container(container_name)
    with open(local_path, "rb") as f:
        c.upload_blob(name=blob_name, data=f, overwrite=True)

def download_file(blob_name: str, local_path: str, container_name: str = AZURE_CONTAINER_NAME):
    c = ensure_container(container_name)
    p = Path(local_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = c.download_blob(blob_name).readall()
    p.write_bytes(data)

