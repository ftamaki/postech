from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    UPLOAD_SERVICE_URL: str = "http://upload-service:8001"
    REPORTS_SERVICE_URL: str = "http://reports-service:8003"

    class Config:
        env_file = ".env"


settings = Settings()
