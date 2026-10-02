"""
工具模块，负责提供 task 相关的辅助能力。
"""

from typing import Dict, List

from .sse_utils import (
    push_to_session,
    SSEEvent
)


# ============================================================
# 内存任务状态
# ============================================================

_tasks_running_list: Dict[str, List[str]] = {}
_tasks_done_list: Dict[str, List[str]] = {}

_tasks_status: Dict[str, str] = {}

_tasks_result: Dict[str, Dict[str, str]] = {}


TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"


# ============================================================
# 节点中文名称
# ============================================================

_NODE_NAME_TO_CN: Dict[str, str] = {

    "upload_file": "开始上传文件",

    "node_entry": "检查文件",

    "node_pdf_to_md": "PDF转Markdown",

    "node_md_img": "Markdown图片处理",

    "node_item_name_recognition": "主体名称识别",

    "node_document_split": "文档切分",

    "node_bge_embedding": "向量生成",

    "node_import_kg": "导入知识图谱",

    "node_import_milvus": "导入向量库",

    "__end__": "处理完成",

    "END": "处理完成",

    # Query流程
    "node_item_name_confirm": "确认问题产品",

    "node_answer_output": "生成答案",

    "node_rerank": "重排序",

    "node_rrf": "倒排融合",

    "node_web_search_mcp": "网络搜索",

    "node_search_embedding": "切片搜索",

    "node_search_embedding_hyde":
        "切片搜索(假设性文档)",

    "node_multi_search": "多路搜索",

    "node_query_kg": "查询知识图谱",

    "node_join": "多路搜索合并",
}


# ============================================================
# 初始化任务
# ============================================================

def _ensure_task(task_id: str) -> None:

    if task_id not in _tasks_running_list:
        _tasks_running_list[task_id] = []

    if task_id not in _tasks_done_list:
        _tasks_done_list[task_id] = []

    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}


def _to_cn(node_name: str) -> str:
    return _NODE_NAME_TO_CN.get(
        node_name,
        node_name
    )


# ============================================================
# running
# ============================================================

def add_running_task(
    task_id: str,
    node_name: str,
    is_stream: bool = False
) -> None:

    _ensure_task(task_id)

    running = _tasks_running_list[task_id]

    if node_name not in running:
        running.append(node_name)

    if is_stream:
        task_push_queue(task_id)


# ============================================================
# done
# ============================================================

def add_done_task(
    task_id: str,
    node_name: str,
    is_stream: bool = False
) -> None:

    _ensure_task(task_id)

    running = _tasks_running_list[task_id]

    _tasks_running_list[task_id] = [
        n
        for n in running
        if n != node_name
    ]

    done = _tasks_done_list[task_id]

    if node_name not in done:
        done.append(node_name)

    if is_stream:
        task_push_queue(task_id)


# ============================================================
# result
# ============================================================

def set_task_result(
    task_id: str,
    key: str,
    value: str
) -> None:

    _ensure_task(task_id)

    _tasks_result[task_id][key] = value


def get_task_result(
    task_id: str,
    key: str,
    default: str = ""
) -> str:

    _ensure_task(task_id)

    return _tasks_result.get(
        task_id,
        {}
    ).get(
        key,
        default
    )


# ============================================================
# status
# ============================================================

def get_task_status(task_id: str) -> str:

    return _tasks_status.get(
        task_id,
        ""
    )


def update_task_status(
    task_id: str,
    status_name: str,
    push_queue: bool = False
) -> None:

    _tasks_status[task_id] = status_name

    if push_queue:
        task_push_queue(task_id)


# ============================================================
# done / running
# ============================================================

def get_done_task_list(
    task_id: str
) -> List[str]:

    _ensure_task(task_id)

    done = _tasks_done_list.get(
        task_id,
        []
    )

    return [
        _to_cn(n)
        for n in done
    ]


def get_running_task_list(
    task_id: str
) -> List[str]:

    _ensure_task(task_id)

    running = _tasks_running_list.get(
        task_id,
        []
    )

    return [
        _to_cn(n)
        for n in running
    ]


# ============================================================
# 把任务进度推给 SSE
# ============================================================

def task_push_queue(task_id: str):

    push_to_session(
        task_id,
        SSEEvent.PROGRESS,
        {
            "status":
                get_task_status(task_id),

            "done_list":
                get_done_task_list(task_id),

            "running_list":
                get_running_task_list(task_id),
        }
    )


# ============================================================
# 清理任务
# ============================================================

def clear_task(task_id: str):

    _tasks_running_list.pop(
        task_id,
        None
    )

    _tasks_done_list.pop(
        task_id,
        None
    )

    _tasks_status.pop(
        task_id,
        None
    )

    _tasks_result.pop(
        task_id,
        None
    )