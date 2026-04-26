from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"
    QUEUE_NAME: str = "diagram.process"
    ANTHROPIC_API_KEY: str = ""
    REPORTS_SERVICE_URL: str = "http://reports-service:8003"
    UPLOAD_SERVICE_URL: str = "http://upload-service:8001"
    UPLOADS_DIR: str = "/uploads"
    # Modelo Claude com suporte a visão e documentos
    CLAUDE_MODEL: str = "claude-sonnet-4-6"

    class Config:
        env_file = ".env"


settings = Settings()
