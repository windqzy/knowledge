import copy
import json
from typing import TypedDict


"""
@dataclass 初始化方法
BaseModel 初始化方法 json处理
TypedDict return {}
"""
class ImportGraphState(TypedDict):
    task_id:str #标记任务的唯一标识
    local_file_path: str|None #标记书收入的原文件地址(不确定类型)
    md_path: str|None #明确的md地址/后续处理完图片地址
    pdf_path:str|None #明确的pdf地址

    file_title:str #文件名 xx.md
    local_dir:str #输出文件的文件夹地址

    md_content:str #md的内容

    is_md_read_enabled: bool #是否是md文件
    is_pdf_read_enabled: bool #是否是pdf文件

    chunk:list[dict]  #切块的内容(还没有向量)
    item_name:str #每个文档提取的唯一标识 多个文档可能相同(多个文档可能相同 比如烫金机的维修/销售。/售后手册)

    embedding_content:list[dict] #切块的内容(有向量)

#创建一个默认对象
graph_default_state: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": [],
}

#更改state对象中的某些值
def create_default_state(**kwargs) -> ImportGraphState:
    """
    更改state对象中的某些值
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """
    state = copy.deepcopy(graph_default_state)
    state.update(kwargs)
    return state

# state = create_default_state(task_id='007',local_file_path='./中文.md')
# print(state)
# print(json.dumps(state,indent=4,ensure_ascii=False))
# state1 = create_default_state(local_dir='./output')
# print(json.dumps(state1,indent=4,ensure_ascii=False))

# 创建一个默认对象
def get_default_state(**kwargs) -> ImportGraphState:
    state = copy.deepcopy(graph_default_state)
    return state
#langgraph ->state->invoke(state)
