import asyncio
import time
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from wrapper import AsyncChatModelWrapper
from chatglm2 import ChatGLM2


app = FastAPI()
chat_model = ChatGLM2()
chat_model_wrapper = AsyncChatModelWrapper(chat_model, gpu_timeout=15)  # 创建AsyncChatModelWrapper的实例
chat_model_wrapper.start()


# 后台任务函数，用于处理异步任务
async def process_task_wrapper():
    await chat_model_wrapper.task_consumer()


# 添加任务的API接口
@app.post("/add_task/")
async def add_task_api(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    task_key = data['task_key']
    task_input = data['task_input']
    chat_model_wrapper.add_task(task_key, task_input)
    background_tasks.add_task(process_task_wrapper)
    return {"message": "Task added to the queue."}


# 获取任务结果的API接口
@app.get("/get_result/")
async def get_result_api(task_key: str):
    result = chat_model_wrapper.get_result(task_key)
    if result:
        return {"task_key": task_key, "result": result[0], "is_ready": result[1]}
    else:
        return {"message": "Task key not found."}


# async def run_chat_model_wrapper():
#     await chat_model_wrapper.start_forever()


# 启动FastAPI应用程序
if __name__ == "__main__":

    # 启动后台任务运行
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(run_chat_model_wrapper())

    uvicorn.run(app, host="0.0.0.0", port=7777)
