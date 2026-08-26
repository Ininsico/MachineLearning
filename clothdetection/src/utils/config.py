import yaml
from pathlib import Path


class Config:
    _instance = None

    def __new__(cls, config_path=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = None
            cls._instance._config_path = config_path
        return cls._instance

    def load(self, config_path: str = None):
        path = config_path or self._config_path or "config.yaml"
        with open(path) as f:
            self._config = yaml.safe_load(f)
        self._config_path = path
        return self._config

    def get(self, *keys, default=None):
        if self._config is None:
            self.load()
        val = self._config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val

    def get_path(self, *keys):
        return Path(self.get(*keys))


cfg = Config()
