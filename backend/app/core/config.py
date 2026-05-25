from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    debug: bool = False
    cors_origins: str = "*"

    class Config:
        env_file = ".env"


settings = Settings()
