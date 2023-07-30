import time

from redis import Redis
from rq import Queue
from rq.job import Job

from chatglm2 import ChatGLM2, BaseChatModel, ChatGLM2TaskInput

models: dict[str, BaseChatModel] = {
    "chat_glm2": ChatGLM2(),
    "chat_glm3": ChatGLM2(),
}

redis_conn = Redis(
    host='10.233.104.50',
    password='corgiclub666'
)

# Create RQ queues for each model
queues: dict[str, Queue] = {
    model_name: Queue(model_name, connection=redis_conn) for model_name in models
}


def enqueue_chat_task(model_name: str, task_input):
    return models[model_name].chat(task_input)


def enqueue_release_gpu(model_name: str):
    return models[model_name].to_cpu()


def register_release_gpu(model_name: str, job_id: str):
    models[model_name].gpu_release_job_id = job_id


def test_job(model_name="chat_glm2"):
    from datetime import timedelta
    queue = queues[model_name]
    task = {
        "query": "你好"
    }
    # 取消之前的显存释放任务
    last_gpu_release_job_id = models[model_name].gpu_release_job_id
    if last_gpu_release_job_id is not None:
        job = queues[model_name].fetch_job(last_gpu_release_job_id)
        if job.get_status() == "queued":
            job.cancel()
            print(f"Job {last_gpu_release_job_id} cancelled")

    chat_job = queue.enqueue(
        enqueue_chat_task, model_name, task,
        # on_success=report_success,
    )

    # gpu_release_delay = 15
    #
    # # 添加显存释放任务
    # release_job = queue.enqueue_in(timedelta(seconds=gpu_release_delay), enqueue_release_gpu)
    # register_release_gpu(model_name, release_job.id)

    # print("job_id", chat_job.id, release_job.id)

    # print("job_result", chat_job.perform())
    # time.sleep(30)
    print("job_result", chat_job.latest_result())

    # job = Job.fetch(id='my_id', connection=redis_conn)
    # print(job.return_value())

    # print("job_result", chat_job.latest_result().return_value)
    # print("job_result", queue.fetch_job(chat_job.id))


if __name__ == "__main__":
    task = {
        "query": "你好"
    }

    aaa = enqueue_chat_task("chat_glm2", task)
    print(aaa)

    task = {
        "query": "你好啊"
    }

    bbb = enqueue_chat_task("chat_glm3", task)
    print(bbb)
