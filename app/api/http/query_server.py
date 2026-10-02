import json
from mimetypes import guess_type
from typing import Any

import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse

from app.api.schemas.query_shema import QueryRequestSchema, SyncQueryResponseSchema, AsyncQueryResponseSchema
from app.process.query.agent.main_graph import query_graph_app
from app.process.query.agent.state import QueryGraphState, create_query_default_state
from app.shared.runtime.logger import logger, PROJECT_ROOT
from app.shared.utils.sse_utils import sse_generator, get_sse_queue, create_sse_queue
from datetime import datetime
from fastapi.requests import Request

from app.shared.utils.task_utils import get_done_task_list, update_task_status, TASK_STATUS_PROCESSING, \
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED

app = FastAPI()


# 接口1：健康状态检查
@app.get('/health')
def health():
    logger.info(f'{datetime.now()}进行健康状态检查服务！')
    return {
        'code': 0
    }


# 接口2:返回chat.html页面
@app.get('/html')
def return_html():
    html_path_obj = PROJECT_ROOT / 'app' / 'resources' / 'htmls' / 'chat.html'
    return FileResponse(path=str(html_path_obj),
                        media_type=guess_type(html_path_obj.name)[0])


# 接口3:流式响应接口
@app.get('/stream/{session_id}')
async def stream(session_id: str, request: Request):
    # 防御性处理：
    # 如果 /stream 比 /query 先访问，
    # 这里也能创建 Queue
    if get_sse_queue(session_id) is None:
        create_sse_queue(session_id)
    logger.info(f'{session_id}建立了流式响应通道！')
    return StreamingResponse(sse_generator(session_id, request), media_type='text/event-stream')


# 同步函数，执行查询图的流程
def invoke_query_graph(original_query: str, session_id: str, is_stream: bool) -> dict[str, Any]:
    try:
        update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
        # 1.创建图需要的state参数
        state: QueryGraphState = create_query_default_state(
            session_id=session_id,
            original_query=original_query,
            is_stream=is_stream
        )
        logger.info(f"开始测试查询图流程：传入参数为：\n{json.dumps(state, indent=4, ensure_ascii=False)}")
        # 2.调用图对象
        result = query_graph_app.invoke(state)
        logger.info(f"测试结束查询图流程：查询结果为：\n{json.dumps(result, indent=4, ensure_ascii=False)}")
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
        return result
    except Exception as e:
        logger.exception(f'查询图执行出现错误！{e}')
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)


# 接口4:查询问题接口
@app.post('/query')
async def query_question(task: BackgroundTasks, param: QueryRequestSchema):
    # 1.获取参数
    session_id = param.session_id
    query: str = param.query
    is_stream = param.is_stream
    # 2.判断是否是流式
    if is_stream:
        # 3.流式的异步执行
        if get_sse_queue(session_id) is None:
            create_sse_queue(session_id)

        task.add_task(invoke_query_graph, original_query=query, session_id=session_id, is_stream=is_stream)
        return AsyncQueryResponseSchema(
            message=f'{query} 问题正在查询和处理中',
            session_id=session_id,
        )
    else:
        # 4.非流式的同步执行
        result: QueryGraphState = invoke_query_graph(original_query=query, session_id=session_id, is_stream=is_stream)
        done_list = get_done_task_list(session_id)
        return SyncQueryResponseSchema(
            message=f'{query} 问题，查询成功！',
            session_id=session_id,
            answer=result.get('answer'),
            done_list=done_list,
            image_urls=result.get('image_urls')
        )


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8001)


"""
health()              → def ✅
return_html()         → def ✅
stream()              → async def ✅【需要改】
invoke_query_graph()  → def ✅
query_question()      → async def ✅【你已经改了】
"""