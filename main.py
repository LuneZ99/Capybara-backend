import asyncio
import queue

import time

from typing import List
from fastapi import FastAPI, Request, Response, status, BackgroundTasks, HTTPException
from json import JSONDecodeError
from fastapi.responses import StreamingResponse
from fastapi.exceptions import RequestValidationError


app = FastAPI()
task_queue = []
task_results = {}


class ComputeModel():

    def __init__(self):
        self.mutex = False

    async def __call__(self, inp):

        while True:
            if not self.mutex:
                self.mutex = True
                await asyncio.sleep(inp)
                self.mutex = False
                print(f"Compute {inp} -> {inp**2}")
                return inp ** 2
            else:
                print(f"pending... {inp}")
                await asyncio.sleep(0.5)




cm = ComputeModel()



async def process_task(task_id: int, input_data: int):
    # 模拟机器学习模型计算的时间（这里使用 sleep 代替）
    # await asyncio.sleep(5)
    # 假设模型计算结果是输入数据的平方
    result = await cm(input_data)
    task_results[task_id] = result


async def get_task_result(task_id: int):
    while task_id not in task_results:
        await asyncio.sleep(1)
    return task_results[task_id]


# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request: Request, exc: RequestValidationError):
#     return Response(content="Invalid input data", status_code=status.HTTP_400_BAD_REQUEST)


@app.post("/submit_task/{task_id}", status_code=status.HTTP_202_ACCEPTED)
async def submit_task(task_id: int, background_tasks: BackgroundTasks, request: Request):
    try:
        inp_data = await request.json()
        print(inp_data)
    except JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")

    task_queue.append((task_id, inp_data['inp']))
    background_tasks.add_task(process_task, task_id, inp_data['inp'])  # 将任务添加到后台任务队列
    return {"message": "Task accepted", "queued_tasks": len(task_queue)}


@app.get("/task_status/{task_id}", status_code=status.HTTP_200_OK)
async def get_task_status(task_id: int):
    if task_id in task_results:
        return {"status": "completed", "result": task_results[task_id]}
    elif any(t_id == task_id for t_id, _ in task_queue):
        return {"status": "queued", "position": next(i for i, (t_id, _) in enumerate(task_queue) if t_id == task_id)}
    else:
        return {"status": "unknown"}


async def generate_sse(task_id: int):
    while task_id not in task_results:
        await asyncio.sleep(1)
        event_data = {"status": "queued", "queued_tasks": len(task_queue)}
        yield f"data: {event_data}\n\n"

    event_data = {"status": "completed", "result": task_results[task_id]}
    yield f"data: {event_data}\n\n"


@app.get("/task_status_stream/{task_id}", status_code=status.HTTP_200_OK)
async def get_task_status_stream(task_id: int):
    response = StreamingResponse(generate_sse(task_id), media_type="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/models_info")
async def models_info():
    return {
        "models": [
            {
                "name": "ChatGLM2-6B",
                "free_level": 0,
                "cost": 0,
                "queue": 0,
                "available": True,
                "default_hyper_params": {
                    "T": 1.0,
                    "max_token": 1024
                }
            },
            {
                "name": "GPT-4 api",
                "free_level": 10,
                "cost": 5,
                "queue": 0,
                "available": True,
                "default_hyper_params": {
                }
            }
        ],
        "device_info": [
            {
                "device_name": "GPU0",
                "device_type": "GPU",
                "total_memory": 24 * 1024 * 1024 * 1024
            },
            {
                "device_name": "GPU1",
                "device_type": "GPU",
                "total_memory": 24 * 1024 * 1024 * 1024
            }
        ]
    }
