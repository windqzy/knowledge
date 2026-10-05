import json
import sys
import time

from app.shared.runtime.logger import node_log
from app.rag.query.item_name_confirm_service import confirm_item_name
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_item_name_confirm")
def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    输入：state['original_query']
    输出：更新 state['item_names']
    """
    # 先登记节点开始，前端进度区可以立即感知"主体确认"已启动。
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    # 调用 rag/query service 层
    state = confirm_item_name(state)
    time.sleep(3)
    # 识别完成后写入完成列表，方便前端展示当前节点已结束。
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    return state

if __name__ == "__main__":
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "HAK 180 烫金机怎么用？",
        "is_stream": False,
    }
    result_state = node_item_name_confirm(mock_state)
    print(result_state)
"""
                用户问题
                   ↓
              获取近期历史
                   ↓
          Query Understanding
          ┌────────┴────────┐
          ↓                 ↓
      问题重写          实体候选提取
          │                 │
          └────────┬────────┘
                   ↓
            Entity Resolution
                   ↓
        ┌──────────┴──────────┐
        ↓                     ↓
   精确/别名命中             未命中
        ↓                     ↓
直接 canonical item     Dense + Sparse/BM25
                              ↓
                         Hybrid Search
                              ↓
                        TopK候选实体
                              ↓
                     Confidence判断
                  ┌───────────┼───────────┐
                  ↓           ↓           ↓
               确定         模糊        不需要实体
                  ↓           ↓           ↓
             item_names    反问用户     通用检索
                  ↓
              后续RAG

1. 取最近历史
2. LLM做 query rewrite + candidate entities
3. 先 exact / alias
4. 不确定再 hybrid search
5. 用 Top1 + margin 做 confidence
6. 确定 → state.item_names
7. 模糊 → state.answer 反问
8. 不需要主体 → 允许继续通用检索
9. 保存本次 user message

老师的整体方向是对的：LLM 理解问题 → Milvus 标准化主体 → 多路召回 → 融合 → 重排 → 回答。
真正要优化的是：历史筛选别太死、item_name 先规则后向量、置信度别只看一个分数、不是所有问题都强制绑定 item_name。
"""