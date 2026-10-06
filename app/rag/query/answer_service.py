import re

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import SUPPORTED_IMAGE_EXTENSIONS
from app.shared.clients import save_chat_message, get_recent_messages
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.task_utils import add_done_task, add_running_task, push_to_session
from app.shared.utils.sse_utils import SSEEvent
from app.shared.runtime.logger import logger, step_log
import time
import sys


# 1. 判断有没有answer
@step_log("has_answer_in_state")
def has_answer_in_state(state) -> bool:
    answer = state.get("answer")
    if answer:
        logger.info(f"没有明确的item_names,state包含answer内容:{answer}")
        return True
    logger.info(f"有明确的item_names,正常调用模型和识别图片地址!")
    return False


# 2. 没有
# 2.1 获取参数和校验
@step_log("get_data_and_validate")
def get_data_and_validate(state) -> tuple:
    is_stream = state.get("is_stream", False)  # bool True | False  | None
    session_id = state.get("session_id")
    item_names = state.get("item_names")
    rewritten_query = state.get("rewritten_query")
    reranked_docs = state.get("reranked_docs")
    if not session_id or not item_names or not rewritten_query or not reranked_docs:
        logger.error("session_id或者item_names或者rewritten_query或者reranked_docs为空,业务无法继续,提前终止!!")
        raise ValueError("session_id或者item_names或者rewritten_query或者reranked_docs为空,业务无法继续,提前终止!!")
    return is_stream, session_id, item_names, rewritten_query, reranked_docs


def _get_valid_history_messages(session_id) -> list[dict]:
    # 2.1 近10条聊天记录,调用history_utils查询对应session_id和limit聊天数据 -> find({session_id:session_id}).order({ts:-1}).limit(10)
    history_messages: list[dict] = get_recent_messages(session_id, limit=10)
    # 2.2 清洗数据,获取有效的聊天记录! item_names不为空!
    valid_history_messages: list[dict] = [message for message in history_messages if message.get("item_names")]
    # 2.3 返回有效的聊天列表
    return valid_history_messages


# 2.2 拼接模型反馈的提示词字符串
@step_log("create_model_prompt")
def create_model_prompt(reranked_docs: list[dict], session_id: str, item_names: list[str], rewritten_query: str) -> str:
    # context -> 主要答案参考内容
    # context -> reranked_docs {chunk_id,file_title,item_name,parent_title,title,part,content,score,type,url}
    # 确定属性:  file_title item_name content score -> reranker模型的分 type 本地数据库 / 网络搜索
    # 确定格式:  参考切片: 1 , 来自文档: file_title ,关联商品/实体: item_name , 置信度:score , 来源: 本地数据库/网络搜索 \n
    #           切片内容: content \n\n
    context: str = ""
    for index, chunk in enumerate(reranked_docs, start=1):
        context += (
            f"参考切片: {index}, 来自文档: {chunk.get('file_title')} ,关联商品/实体: {chunk.get('item_name')} , "
            f"置信度:{chunk.get('score')} , 来源:{'本地数据库' if chunk.get('type') == 'milvus' else '网络搜索'}\n"
            f" 切片内容: {chunk.get('content')} \n\n")

    # history [查询有效的聊天记录]
    history_text: str = ""
    valid_history_messages = _get_valid_history_messages(session_id)
    if valid_history_messages:
        # role = user  text = 原始问题  rewritten_query = 重写的问题  item_names = [1,2,3] image_urls = []
        # 本次为提问,原始问题为:xx,上一次问题重写后:xxx,问题管理的item_name:x,x,x
        # role = assistant  text = 模型的回答  rewritten_query = 重写的问题  item_names = [1,2,3] image_urls
        # 本次关于: 重写问题 的回答为:xxx,关联的实体为:x,x,x
        # 提取item_names / 重写问题
        for message in valid_history_messages:
            if message.get("role") == "user":
                history_text += f"本次是提问记录,原始问题为:{message.get('text')},问题重写后:{message.get('rewritten_query')},问题关联的item_name:{message.get('item_names')}\n"
            else:
                history_text += f"本次是回答记录,关于问题:{message.get('rewritten_query')}的回答! 回答内容为:{message.get('text')[:100]}....,关联的item_name:{message.get('item_names')}\n"
    else:
        history_text = "无有效的聊天记录!"

    prompt_text: str = load_prompt("answer_out", context=context, history=history_text, item_names=item_names,
                                   question=rewritten_query)
    return prompt_text


# 2.3 调用模型获取回答的answer
@step_log("call_model_create_answer")
def call_model_create_answer(prompt_text: str, is_stream, session_id: str) -> str:
    # 初始化模型客户端
    llm_model = llm_provider.chat()
    # 初始化message
    messages = [
        HumanMessage(content=prompt_text)
    ]
    # 初始化调用链接
    chains = llm_model | StrOutputParser()
    answer: str = ""
    # 判断是否是流式
    if is_stream:
        # 流式执行  delta | answer
        stream = chains.stream(messages)
        for content in stream:
            # 1 2 3 4 5  1 2 3  4 5
            # 前端推送数据 delta
            # 收集answer
            answer += content
            push_to_session(session_id, SSEEvent.DELTA, {"delta": content})
    else:
        # 非流式执行 answer
        answer = chains.invoke(messages)
        # 获取answer
    return answer


# 2.4 提供物料reranked_docs获取图片链接 -> image_urls
@step_log("extract_image_urls_in_docs")
def extract_image_urls_in_docs(reranked_docs: list[dict]) -> list[str]:
    #  reranked_docs {chunk_id,file_title,item_name,parent_title,title,part,content,score,type,url}
    #  content -> md ![]() ![](http)  url-> url html / 图片
    #  url是不是图片,后缀名
    #  content包含图片正则:  r"\!\[.*?\]\((.*?)\)" -> finditer search match sub findall
    image_urls: list[str] = []
    image_re = re.compile(r"\!\[.*?\]\((.*?)\)")

    for chunk in reranked_docs:
        url: str = chunk.get("url", "")
        content = chunk.get("content")
        if url.endswith(SUPPORTED_IMAGE_EXTENSIONS):
            image_urls.append(url)
        content_image_urls: list[str] = image_re.findall(content)
        if content_image_urls:
            image_urls.extend(content_image_urls)
    logger.info(f"已经完成图片识别,具体的数量:{len(image_urls)}")
    return image_urls


def generate_answer(state: QueryGraphState) -> QueryGraphState:
    # 1. 判断有没有answer
    has_answer:bool = has_answer_in_state(state)
    if not has_answer:
       #2. 没有
       #2.1 获取参数和校验
       is_stream,session_id,item_names,rewritten_query,reranked_docs = get_data_and_validate(state)
       #2.2 拼接模型反馈的提示词字符串
       prompt_text:str = create_model_prompt(reranked_docs,session_id,item_names,rewritten_query)
       # 2.3 调用模型获取回答的answer
       answer:str = call_model_create_answer(prompt_text,is_stream,session_id)
       # 2.4 提供物料reranked_docs获取图片链接 -> image_urls
       image_urls:list[str] = extract_image_urls_in_docs(reranked_docs)
       #2.5 更新state <- answer | image_urls
       state['answer'] = answer
       state['image_urls'] = image_urls
    #3. 有
    #4. 更新聊天记录[assistant]
    save_chat_message(
        session_id=state.get("session_id"),
        role="assistant",  # 提问
        text=state.get("answer"),  # 前端展示
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names",[]),
        image_urls=state.get("image_urls",[])
    )
    logger.info(f"完結撒花~~2026年8月19日16:01:10")
    return state