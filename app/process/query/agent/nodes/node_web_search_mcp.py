import sys

from app.shared.runtime.logger import node_log
from app.rag.query.web_search_service import search_by_web
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_web_search_mcp")
def node_web_search_mcp(state):
    """
    节点功能：调用外部搜索引擎补充信息
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    state = search_by_web(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    return {
        'web_search_docs':[]
    }
    #并发的地方不能有相同的key，有就报错