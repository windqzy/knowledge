from typing import Any

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNS_SPLIT_BATCH_SISE
from app.shared.runtime.logger import step_log
from app.shared.runtime.logger import logger
from app.infra.llm.providers import generate_embeddings


# 1. 获取参数并且校验
@step_log("get_data_and_validate")
def get_data_and_validate(state: ImportGraphState) -> list[dict[str, Any]]:
    chunks = state.get("chunks")
    if not chunks:
        logger.error("chunks为空，业务无法继续，提前终止！")
        raise ValueError("chunks为空，业务无法继续，提前终止！")
    # 返回数据
    return chunks


# 2. 批量生成向量
@step_log("batch_create_chunks_vector")
def batch_create_chunks_vector(chunks: list[dict[str, Any]]):
    # 每五个一批批量生成
    chunk_size = len(chunks)
    # 批量循环
    for index in range(0, chunk_size, 5):
        current_chunks = chunks[index:index + CHUNS_SPLIT_BATCH_SISE]
        current_content_list: list[str] = [f'{chunk.get('item_name')}:{chunk.get('content')}' for chunk in chunks]
        result = generate_embeddings(current_content_list)

        for index, chunk in enumerate(current_chunks):
            chunk['dense_vector'] = result['dense'][index]
            chunk['sparse_vector'] = result['sparse'][index]
        logger.debug(f'前{index%CHUNS_SPLIT_BATCH_SISE+1}批数据处理完毕，本次处理长度：{len(current_chunks)}')


@step_log("generate_chunk_embeddings")
def generate_chunk_embeddings(state: ImportGraphState) -> ImportGraphState:
    # 1. 获取参数并且校验
    chunks = get_data_and_validate(state)
    # 2. 批量生成向量
    batch_create_chunks_vector(chunks)
    # 3. 更新state并返回..
    state['embeddings_content'] = chunks
    return state
