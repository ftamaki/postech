from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/upload_db"
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"
    UPLOADS_DIR: str = "/uploads"
    QUEUE_NAME: str = "diagram.process"

    class Config:
        env_file = ".env"


settings = Settings()
