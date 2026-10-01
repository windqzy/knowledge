import json
import shutil
import uuid
from mimetypes import guess_type
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse

from app.api.schemas.import_schema import StatusResponseSchema, UploadResponseSchema
from app.process.import_.agent.main_graph import import_graph_app
from app.process.import_.agent.state import ImportGraphState, create_default_state
from app.shared.runtime.logger import logger, PROJECT_ROOT
from app.shared.utils.task_utils import get_running_task_list, get_done_task_list, get_task_status, update_task_status, \
    TASK_STATUS_FAILED, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED
from datetime import datetime

app = FastAPI()


# 接口1:返回html文件
@app.get("/html")
def return_html():
    # 拼接html地址
    html_path_obj: Path = PROJECT_ROOT / 'app' / 'resources' / 'htmls' / 'import.html'
    # 返回响应File
    return FileResponse(path=str(html_path_obj),
                        media_type=guess_type(html_path_obj.name)[0])


# 接口2:返回指定task_id的任务状态
@app.get("/status/{task_id}")
def return_status(task_id: str):
    # 任务状态/运行列表/完成列表-task_utils
    running_list = get_running_task_list(task_id)
    done_list = get_done_task_list(task_id)
    status = get_task_status(task_id)

    return StatusResponseSchema(
        code=200,
        task_id=task_id,
        status=status,
        done_list=done_list,
        running_list=running_list
    )


def invoke_import_graph(task_id: str, local_file_path: str, local_dir: str):
    state: ImportGraphState = create_default_state(
        # local_file_path:str
        local_file_path=local_file_path,
        local_dir=local_dir,
        task_id=task_id
    )
    logger.info(f"开始流程测试,测试数据:{json.dumps(state, indent=4, ensure_ascii=False)}")
    try:
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        # 图
        result: ImportGraphState = import_graph_app.invoke(state)
        update_task_status(task_id, TASK_STATUS_COMPLETED)
    except Exception as e:
        update_task_status(task_id, TASK_STATUS_FAILED)
        logger.exception(e)


# 接口3:上传文件+图执行
@app.post("/upload")
async def upload_files(task: BackgroundTasks, files: list[UploadFile]):
    # 1.本次文件生成一个唯一的id
    task_id = str(uuid.uuid4())
    # 2.转存文件 /output/时间/task_id/filename
    uploaded_file = files[0]
    local_dir_obj: Path = PROJECT_ROOT / 'output' / datetime.now().strftime("%Y%m%d") / task_id
    local_dir_obj.mkdir(parents=True, exist_ok=True)
    local_file_path_obj: Path = local_dir_obj / uploaded_file.filename
    # data = await uploaded_file.read()
    # local_file_path_obj.write_bytes(data=data)
    with open(local_file_path_obj, "wb") as f:
        shutil.copyfileobj(uploaded_file.file, f)
    # 3.异步调用图的执行流程
    task.add_task(
        invoke_import_graph,
        task_id=task_id,
        local_file_path=str(local_file_path_obj),
        local_dir=str(local_dir_obj)
    )
    # 4.返回结果
    return UploadResponseSchema(
        code=200,
        message=f'{uploaded_file.filename}已经开始进行解析了！',
        task_ids=[task_id]
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
① /upload
上传文件 + 启动后台任务 + 返回 task_id

② LangGraph
后台一步一步跑

③ /status/{task_id}
前端查询现在跑到哪了

一句话记：
task_id 就像快递单号；上传接口负责“下单”，LangGraph负责“运输”，status接口负责“查物流”。
"""
