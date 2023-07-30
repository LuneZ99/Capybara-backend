from pathlib import Path
from typing import Any, Optional

import torch

cache_folder = "/mnt/hdd-2/transformers/cache/"
proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890',
}


class GPUNotAvailableError(Exception):
    def __init__(self, message="There are currently no GPU devices available."):
        self.message = message
        super().__init__(self.message)


class BaseChatModel:
    model_name: str
    display_name: str
    model: Optional[Any]
    gpu_memory_usage: Optional[int]
    gpu_release_job_id: Optional[str]

    def __init__(self, gpu_memory_usage=0):
        self.cache_folder = Path(cache_folder)
        self.proxies = proxies
        self.gpu_memory_usage = gpu_memory_usage
        self.gpu_release_job_id = None
        self.is_model_available = False

    def chat(self, task_input):
        raise NotImplementedError

    def stream_chat(self, task_input):
        print(f"This model ({self.model_name}) does not support streaming output, fallback to normal chat function")
        return self.chat(task_input)

    def get_model_info(self):
        return {
            "info": "Base model for all chat models."
        }

    def _load_model(self, device):
        raise NotImplementedError

    def to_gpu(self, device=None):
        print(f"move {self.model_name} to gpu")
        if self.is_model_available:
            if self.model.device == torch.device("cpu"):
                self._cpu_to_gpu(device)
        else:
            self._disk_to_gpu(device)

    def to_disk(self):
        print(f"move {self.model_name} to disk")
        self.model = None
        self.is_model_available = False

    def to_cpu(self):
        print(f"move {self.model_name} to cpu")
        if self.is_model_available:
            if self.model.device != torch.device("cpu"):
                self._gpu_to_cpu()
        else:
            self._disk_to_cpu()

    def _disk_to_cpu(self):
        self._load_model(torch.device("cpu"))
        self.is_model_available = True

    def _disk_to_gpu(self, device=None):
        if device is None:
            device = self.get_first_available_gpu()
        if device:
            self._load_model(device)
            self.is_model_available = True
        else:
            raise GPUNotAvailableError("There are currently no GPU devices available.")

    def _gpu_to_cpu(self):
        self.model = self.model.to(torch.device("cpu"))

    def _cpu_to_gpu(self, device=None):
        if device is None:
            device = self.get_first_available_gpu()
        if device:
            self.model = self.model.to(device)
        else:
            raise GPUNotAvailableError("There are currently no GPU devices available.")

    def total_gpu_memory(self, device=None):
        if device is None and self.model is not None:
            device = self.model.device
        return torch.cuda.get_device_properties(device=device).total_memory

    def using_gpu_memory(self, device=None):
        if device is None and self.model is not None:
            device = self.model.device
        return torch.cuda.max_memory_allocated(device=device)

    def remain_gpu_memory(self, device=None):
        if device is None and self.model is not None:
            device = self.model.device
        return self.total_gpu_memory(device) - self.using_gpu_memory(device)

    def get_first_available_gpu(self):
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for _id in range(gpu_count):
                device = torch.device(f"cuda:{_id}")
                if self.gpu_memory_usage < self.remain_gpu_memory(device):
                    print(f"{self.model_name} load to {device}")
                    return device
        else:
            return None

    # torch.cuda.memory_usage
    # torch.cuda.OutOfMemoryError
