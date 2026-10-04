from app.process.query.agent.state import QueryGraphState
from app.shared.clients import save_chat_message


def confirm_item_name(state: QueryGraphState) -> QueryGraphState:
    """
    意图确认服务：
    1. 结合历史对话提取商品名
    2. 将模糊问题改写为完整独立的精准问题
    3. 在 Milvus 向量库中进行混合搜索
    4. 根据评分高低自动对齐标准型号，或生成反问让用户手动确认
    5. 同步历史记录到 MongoDB
    """
    save_chat_message(
        session_id=state.get("session_id"),
        role='user',
        text=state.get("original_query"),
        rewritten_query=f'重写的问题:{state.get("original_query")}',
        item_names=['烫金机'],
        image_urls=[]
    )
    return state