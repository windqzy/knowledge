import sys

from app.shared.runtime.logger import node_log
from app.rag.query.rerank_service import rerank_documents
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_rerank")
def node_rerank(state):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    state = rerank_documents(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state