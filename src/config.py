from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    redis_url: str = "redis://redis:6379"
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    batch_size: int = 32
    allowed_origins: str = "*"
    allowed_hosts: str = "*"

    model_config = ConfigDict(env_file=".env")

settings = Settings()
