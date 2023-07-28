from pathlib import Path
from typing import Any


class BaseChatModel:
    model_name: str
    model: Any

    def __init__(self):
        self.cache_folder = Path("/mnt/hdd-2/transformers/cache/")
        self.proxies ={
            'http': 'http://127.0.0.1:7890',
            'https': 'http://127.0.0.1:7890',
        }

    def chat(self, task_input):
        raise NotImplementedError

    def stream_chat(self, task_input):
        raise NotImplementedError

    def get_model_info(self):
        return {
            "info": "not implemented."
        }

    def to(self, *args, **kwargs):
        self.model = self.model.to(*args, **kwargs)
        return self
