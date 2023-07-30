from fastapi import FastAPI, Depends, Request
from rqueue import *
from datetime import timedelta


app = FastAPI()


@app.post("/chat/{model_name}")
async def chat(model_name: str, request: Request, gpu_release_delay: int = 300):

    queue = queues[model_name]
    task_input = await request.json()['task_input']

    # 取消之前的显存释放任务
    last_gpu_release_job_id = models[model_name].gpu_release_job_id
    if last_gpu_release_job_id is not None:
        job = queues[model_name].fetch_job(last_gpu_release_job_id)
        if job.get_status() == "queued":
            job.cancel()

    # 添加模型推理任务
    chat_job = queue.enqueue(enqueue_chat_task, model_name, task_input)

    # 添加显存释放任务
    release_job = queue.enqueue_in(timedelta(seconds=gpu_release_delay), enqueue_release_gpu)
    register_release_gpu(model_name, release_job.id)

    return {
        "job_id": chat_job.id
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7777)
