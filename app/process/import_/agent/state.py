import json
from typing import TypedDict
import copy

"""
 @dataclass -> 初始化方法 配置类和实体类
 BaseModel -> 初始化方法 严格模式 json处理 -> fastapi
 TypedDict -> langgraph  -> return {}
"""

class ImportGraphState(TypedDict):
    task_id:str                 # 标记任务的唯一标识

    local_file_path:str         # 标记输入的原文件地址(不确定类型)
    md_path:str|None                 # 明确的md地址 / 后续处理完图片地址
    pdf_path:str|None              # 明确的pdf的地址
    file_title:str              # 文件名 xx.md -> xx
    local_dir:str               # 输出文件的文件夹地址

    md_content:str              # md的内容

    is_md_read_enabled:bool     # 是md文件
    is_pdf_read_enabled:bool    # 是pdf文件

    chunks:list[dict]            # 切块的内容(没有向量)

    item_name:str               # 每个文档提取的唯一标识! 多个文档可能相同(烫金机维修/销售/售后手册)

    embeddings_content:list[dict] # 切块的内容(有向量)

# 默认对象 模板
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


# 创建指定参数的对象
def create_default_state(**kwargs) -> ImportGraphState:
    """
      创建对象可以修改参数
    :param **kwargs 传入:key -> 修改属性名 =value 修改的属性值 , key=value   接收: 字典 {key:value,key:value}
    :return:
    """

    # dict -> update -> {key:value} -> 修改原有值对应的key
    """
      深拷贝: 创建对象,会复制对象有嵌套的属性  适合有集合或者元组这类的嵌套属性  全新的对象 = copy.deepcopy(对象)
      浅拷贝: 创建对象,只会复制对象没有嵌套的属性  适合没有集合或者元组这类的嵌套属性  对象.copy() / copy.copy(对象) 
    """
    state = copy.deepcopy(graph_default_state)
    state.update(kwargs)
    return state

# 创建一个默认对象
def get_default_state() -> ImportGraphState:
    state = copy.deepcopy(graph_default_state)
    return state

