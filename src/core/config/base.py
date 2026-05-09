import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass
class BaseConfig:
    symbols: list[str]
    model_path: str
    timeframe: str
    htf_timeframe: str
    _env_loaded: ClassVar[bool] = False

    @classmethod
    def _load_dotenv(cls) -> None:
        if cls._env_loaded:
            return

        env_path = Path.cwd() / ".env"
        if not env_path.exists():
            cls._env_loaded = True
            return

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key:
                os.environ[key] = value

        cls._env_loaded = True

    @classmethod
    def env_str(cls, key: str, default: str = "") -> str:
        cls._load_dotenv()
        return os.getenv(key, default)

    @classmethod
    def env_list(cls, key: str, default: str = "") -> list[str]:
        raw = cls.env_str(key, default)
        return [item.strip() for item in raw.split(",") if item.strip()]
