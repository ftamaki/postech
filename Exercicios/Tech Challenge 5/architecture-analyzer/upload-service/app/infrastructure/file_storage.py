import logging
import mimetypes
import os

from ..domain.ports import FileStorage as FileStoragePort

logger = logging.getLogger(__name__)


class LocalFileStorage(FileStoragePort):
    def __init__(self, uploads_dir: str):
        self._uploads_dir = uploads_dir
        os.makedirs(uploads_dir, exist_ok=True)

    async def save(self, file_data: bytes, filename: str, content_type: str) -> str:
        extension = mimetypes.guess_extension(content_type) or ""
        # Normaliza extensões comuns
        ext_map = {".jpe": ".jpg", ".jpeg": ".jpg", None: ""}
        extension = ext_map.get(extension, extension)

        safe_filename = f"{filename}{extension}"
        file_path = os.path.join(self._uploads_dir, safe_filename)

        with open(file_path, "wb") as f:
            f.write(file_data)

        logger.info(f"Arquivo salvo: {file_path} ({len(file_data)} bytes)")
        return file_path
