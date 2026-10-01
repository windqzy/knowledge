from app.process.query.agent.state import QueryGraphState


def rerank_documents(state: QueryGraphState) -> QueryGraphState:
    """
    重排序服务：
    1. 合并 RRF 和 Web Search 的文档
    2. 使用 BGE Reranker 模型计算相关性得分
    3. 根据得分动态截断，智能截取 TopK
    4. 回写 reranked_docs
    """
    return state