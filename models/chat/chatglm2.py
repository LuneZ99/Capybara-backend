from typing import Any, Tuple, List

from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

from base import BaseChatModel
import torch


class ChatGLM2TaskInput(BaseModel):
    query: str
    history: List[Tuple[str, str]] = []
    max_length: int = 8192
    num_beams: int = 1
    do_sample: bool = True
    top_p: float = 0.8
    temperature: float = 0.8
    logits_processor: Any = None


class ChatGLM2(BaseChatModel):
    # gpu_memory_usage = 12551120896
    tokenizer: AutoTokenizer

    def __init__(self, model_name="THUDM/chatglm2-6b", display_name=None, gpu_memory_usage=13_000_000_000):
        super().__init__(gpu_memory_usage)

        self.model_name = model_name
        if display_name:
            self.display_name = display_name
        else:
            self.display_name = model_name.split('/')[-1]

    def _load_model(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_folder,
            proxies=self.proxies
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_folder,
            proxies=self.proxies,
            device=device
        ).eval()

    def get_model_info(self):
        pass

    def chat(self, task_input):
        task_input = ChatGLM2TaskInput(**task_input).model_dump()
        self.to_gpu()
        return self.model.chat(self.tokenizer, **task_input)

    def stream_chat(self, task_input):
        task_input = ChatGLM2TaskInput(**task_input).model_dump()
        self.to_gpu()

        current_length = 0
        for new_response, _ in self.model.stream_chat(self.tokenizer, **task_input):
            if len(new_response) == current_length:
                continue
            # new_text = new_response[current_length:]
            current_length = len(new_response)
            yield new_response


if __name__ == '__main__':
    glm = ChatGLM2()
    glm.to_gpu()
    print(glm.chat(
        task_input={"query": "你好"}
    ))
    for x in glm.stream_chat(task_input={"query": "你好"}):
        print(x)

    print(glm.total_gpu_memory())
    print(glm.using_gpu_memory())
