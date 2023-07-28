import asyncio
import time
import cacheout
import torch


class AsyncChatModelWrapper:
    def __init__(self, chat_model, max_tasks=64, gpu_timeout=10 * 60, device="cuda:0"):

        self.chat_model = chat_model
        print(self.chat_model)
        self.queue = asyncio.Queue(maxsize=max_tasks)
        self.results_cache = cacheout.Cache(maxsize=1024)
        self.gpu_timeout = gpu_timeout
        self.device = device

        self.total_execution_time = 0
        self.last_execution_time = time.time()

        self.stop_event = asyncio.Event()
        self.stream = False

        # self.start()

    async def process_task(self, task):

        task_key, task_input = task
        start_time = time.time()
        self.chat_model = self.chat_model.to(self.device)
        print(self.chat_model)

        if self.stream:
            res = None
            for res in self.chat_model.stream_chat(task_input):
                self.results_cache.set(task_key, (res, False))
            self.results_cache.set(task_key, (res, True))
        else:
            result = self.chat_model.chat(task_input)
            self.results_cache.set(task_key, (result, True))

        end_time = time.time()
        execution_time = end_time - start_time

        self.total_execution_time += execution_time
        self.last_execution_time = end_time

    async def task_consumer(self):
        # while not self.stop_event.is_set():
        while True:
            try:
                task = self.queue.get_nowait()
                await self.process_task(task)
                self.queue.task_done()
            except asyncio.QueueEmpty:
                if time.time() - self.last_execution_time > self.gpu_timeout \
                        and self.chat_model.model.device != torch.device('cpu'):
                    print('GPU cache timeout, move model to cpu')
                    self.chat_model = self.chat_model.to('cpu')
                print(f"No task, model device {self.chat_model.model.device}")
                print(f"GPU timeout {time.time() - self.last_execution_time:.2f} s")
                for key, value in self.results_cache.items():
                    print(f'{key}: {value}')
                await asyncio.sleep(0.1)

    def add_task(self, task_key, task_input):
        self.queue.put_nowait((task_key, task_input))
        print(f"Task in queue: {self.queue.qsize() - 1}")

    def get_result(self, task_key):
        return self.results_cache.get(task_key)

    def remove_task(self, task_key):
        # Since asyncio.Queue does not provide a method to remove a specific item,
        # we can just ignore the result when processing the task with the specified key.
        # This will effectively "remove" the task from the queue.
        raise NotImplementedError

    def count_queued_tasks(self):
        return self.queue.qsize()

    def get_avg_execution_time(self):
        return self.total_execution_time / self.queue.qsize()

    # async def start_forever(self):
    #     while True:
    #         await self.task_consumer()

    def start(self):
        # while True:
        #     self.task_consumer()
        asyncio.ensure_future(self.task_consumer())

    def stop(self):
        self.stop_event.set()


# async def test_wrapper():



if __name__ == '__main__':
    from chatglm2 import ChatGLM2

    model = ChatGLM2()

    # 2. 创建异步包装器实例
    async_model_wrapper = AsyncChatModelWrapper(model, device="cuda:1", gpu_timeout=5)
    async_model_wrapper.start()

    # 4. 启动异步处理
    # async_model_wrapper.start()

    # 3. 添加任务到队列
    task1_input = {
        "query": "你好"
    }
    task2_input = {
        "query": "计算一下1+1"
    }
    async_model_wrapper.add_task("task1", task1_input)
    async_model_wrapper.add_task("task2", task2_input)

    task3_input = {
        "query": "你好哈",
        "history": [("哈哈", "哈哈哈")]
    }
    async_model_wrapper.add_task("task3", task3_input)

    print(1)

    async_model_wrapper.start()
