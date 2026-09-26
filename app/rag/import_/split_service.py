import json
import re
from pathlib import Path
from re import split
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MIN, CHUNK_MAX_SIZE
from app.shared.runtime.logger import logger, step_log


# 1. 获取参数和校验
@step_log("validate_data_and_content")
def validate_data_and_content(state: ImportGraphState) -> tuple[str, str, Path]:
    # 1.1 获取参数 md_content / md_path  / file_title
    md_content: str = state.get('md_content')
    file_title: str = state.get('file_title')
    md_path: str = state.get('md_path')
    # 1.2 非空校验 md_path为空不可以的!  file_title md_path读取 md_content  为空从md_path读取
    if not md_path:
        logger.error(f"md_path为空，业务无法继续，提前终止！")
        raise ValueError(f"md_path为空，业务无法继续，提前终止！")
    md_path_obj: Path = Path(md_path)
    if not md_path_obj.is_file():
        logger.error(f"md_path有值：{md_path}，但是没有真实的文件！业务无法继续，提前终止！")
        raise FileNotFoundError(f"md_path有值：{md_path}，但是没有真实的文件！业务无法继续，提前终止！")
    if not md_content:
        logger.warning(f'在state中没有获取到md_content值，利用md_path再次读取！')
        md_content: str = md_path_obj.read_text(encoding='utf-8')
        state['md_content'] = md_content
    if not file_title:
        logger.warning(f'在state中没有获取到file_title值，利用md_path再次读取！')
        file_title: str = md_path_obj.stem
        state['file_title'] = file_title
    # 1.3 统一换行符号  md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    md_content: str = md_content.replace("\r\n", "\n").replace("\r", "\n")
    # 1.4 返回三个参数即可
    return md_content, file_title, md_path_obj


# 2. 根据语义文档的初次切割 [md-标题]
@step_log("split_document_by_title")
def split_document_by_title(md_content: str, file_title: str) -> list[dict[str, Any]]:
    # [chunk{title,file_title,content}]
    # 核心: 按行处理 如果第一个标题,记录,标题下的每一行也记录[], 碰到第二个标题, 将第一个标题+行存储到列表!! 开始记录第二个标题...
    # 2.1 定义个chunks的列表 = []  目标: [chunk [标题+内容] , chunk  [标题+内容] , chunk  [标题+内容]]
    chunks: list[dict[str, Any]] = []
    # 2.2 定义一个当前的标题和当前标题的行存储数据  current_title:str  , current_lines:list[str]  , is_code:bool = False
    current_title: str | None = None  # 当前正在处理的title
    current_lines: list[str] = []  # 当前正在处理的标题对应的行
    is_code: bool = False  # 是否在代码块中
    is_continue_title: bool = False
    # 2.3 md_content按行切割 md_content_lines:list[str] = split(\n)
    md_content_lines: list[str] = md_content.split("\n")
    # 2.4 定义匹配标题的正则规则  可以使用空格开头,后面 1-6#,必须有一个空格,必须是有效标题 内容  r"^\s*#{1,6}\s.+"
    re_line_title = re.compile(r"^\s*#{1,6}\s.+")

    # 2.5 循环处理md切割的数据行 for 行 in md行列表
    for current_line in md_content_lines:
        # 额外判断下空行
        current_line_strip = current_line.strip()
        if not current_line_strip:
            logger.debug(f"当前是{current_title}的空行,跳出处理!!!")
            continue
        # 2.6 判断是否进入和跳出代码块 line.start_with("```" "~~~"):  is_code = not is_code
        # todo current_line_strip此处startswith写成了strip()
        if current_line_strip.startswith("```") or current_line_strip.startswith("~~~"):
            # ```python
            # ```
            is_code = not is_code
            current_lines.append(current_line_strip)
            logger.debug(f"当前是{current_title}的代码块进入或者跳出行!")
            continue
        # 2.7 如果是普通行 -> current_lines
        # 2.8 如果是标题行 & 不在代码块 -> current_title / current_linse  -> chunk {title,file_title,content ="\n",join(current_lines)} -> chunks
        # -> current_title -> 本次行  || current_linse = [标题行]
        if not is_code and re_line_title.match(current_line_strip):
            # 碰到新的标题
            # 当前current_title / current_linse已经变成旧
            # 1. 判断连续标题
            if is_continue_title:
                # 上一次内容存储的是标题! 本次进入到了标题! 连续标题
                # current_lines = [# 标题 , 内容 , 内容 ]
                current_title = current_title + "\n" + current_line_strip
                title_list: list[str] = current_title.split("\n")
                title_list.extend(current_lines[len(title_list) - 1:])
                current_lines = title_list
                # 删除: current_lines.append(current_line_strip)
                # current_title = #标题\n#标题2\n标题3
                logger.debug(f"碰见连续标题,进行了标题和当前行的内容累加!!!")
                continue
            # 2. 孤儿数据  current_title标题为空, current_lines有数据
            if not current_title and current_lines:
                # current_lines <- 累加 - 不结算
                # current_line_strip -> 标题
                # 行 行 行 标题 行 行 行 -> content -> 长切割 -> 标题\n行 行 行   == 标题\n行 行 行
                # 标题在前,内容在后!!
                # current_title = None
                # current_lines = [1,2,3]
                # current_line_strip = #标题1  -> [#标题1 , 1, 2,3]
                temp_lines: list[str] = [current_line_strip]
                temp_lines.extend(current_lines)
                # [#标题 内容 内容]
                current_lines = temp_lines  # [#标题1 , 1, 2,3]
                # current_lines.append(current_line_strip)
                # #标题
                current_title = current_line_strip
                is_continue_title = True
                logger.debug(f"碰到了孤儿数据结算!!!")
                continue
            # 3. 正常的标题和数据  正常 = 之前有标题 = 存储的不是都标题
            # 结算 第一个标题 / 空标题
            if current_lines:
                chunks.append(
                    {
                        "file_title": file_title,
                        "title": current_title,
                        "content": "\n".join(current_lines)
                    }
                )
            is_continue_title = True
            # current_line_strip -> 新的标题
            current_title = current_line_strip  # 新的标题覆盖旧的标题
            current_lines = [current_title]  # 新的行覆盖旧的行
        else:
            # 普通行
            # [内容 内容] current_title = None
            is_continue_title = False
            current_lines.append(current_line_strip)

    # 2.9 循环结束 最后结算一次!!!
    # 情况: 整个文档没有title
    if current_lines:
        chunks.append(
            {
                "file_title": file_title,
                "title": current_title or "default",  # 给与一个默认值
                "content": "\n".join(current_lines)
            }
        )
    # 2.10 返回chunks可以了
    return chunks


@step_log("_too_long_split")
def _too_long_split(chunk):
    sub_chunks: list[dict[str, Any]] = []
    # 1.数据的清洗和处理
    content: str = chunk.get('content')
    title: str = chunk.get('title')
    file_title: str = chunk.get('file_title')
    clean_content: str = content[len(title) + 1:]
    sub_content_prefix = title + '\n'

    # 2.递归切割器
    splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "，",
            "、"
        ],
        chunk_size=CHUNK_SIZE - len(sub_content_prefix),  # 标题 + 正文 <= 600
        chunk_overlap=CHUNK_OVERLAP  # 要>=当前最小可保留片段长度
    )
    """
                   一大段文本
                    │
               按段落切
             ┌──────┴──────┐
           合适            太长
                            │
                         按换行
                            │
                           太长
                            │
                         按句号
                       ┌────┴────┐
                     合适       太长
                                │
                              按逗号
    """

    # 3.使用递归切割器切割即可
    for part, sub_content in enumerate(splitter.split_text(clean_content), start=1):
        sub_chunks.append({
            'title': f'{title}_{part}',
            'parent_title': title,
            'part': part,
            'content': sub_content_prefix + sub_content,
            "file_title": file_title
        })
    logger.info(f"{title}标题对应切换,切成:{len(sub_chunks)}块!")
    return sub_chunks


"""
粗切 Section
       ↓
超过目标尺寸
       ↓
RecursiveCharacterTextSplitter
       ↓
段落 → 行 → 句子 → 分句 → 字符
       ↓
得到很多子 Chunk
       ↓
有些子 Chunk 太短
       ↓
_two_short_merge
       ↓
只允许同语义父标题向后合
       ↓
最终 Chunk
"""


@step_log("_too_short_merge")
def _too_short_merge(refine_chunks):
    """
      实现依据: 1. 前指针 长度小于400  2. 前后是同一个parent_title 且不能为空   3. 合并后长度小于1000
                                -> 触发合并   前 <-- 前 content + 后 content
                                -> 不触发合并  前 -> 添加到一个列表中   后 -> 前
    :param refine_chunks:
    :return:
    """
    logger.info(f"进行短合并处理,进入的数据长度:{len(refine_chunks)}")
    # 1. 定一个装最终chunk的列表
    final_chunks: list[dict[str, Any]] = []
    # 2.定义pre_chunk|base_chunk
    pre_chunk: dict[str, Any] | None = None
    # 3. 循环列表进行合并
    for next_chunk in refine_chunks:
        # 第一次
        if not pre_chunk:
            # 空 -> 第一次循环
            logger.debug(f"短合并第一次进入循环,next_chunk->pre_chunk")
            pre_chunk = next_chunk
            continue
        # 第二开始判断是否需要合并 1. pre_chunk < 400 2. 非空 同一个parent_title 3. 合并后长度小于1000
        # pre_content = parent_title\n 内容
        pre_content: str = pre_chunk.get("content")
        if len(pre_content) < CHUNK_MIN:
            # 可以进行合并
            logger.debug(f'{pre_chunk.get('title')}对应的内容小于{CHUNK_MIN},可以继续检查合并！')
            pre_parent_title = pre_chunk.get("parent_title")
            next_parent_title = next_chunk.get("parent_title")
            if pre_parent_title and (next_parent_title == pre_parent_title):
                logger.debug(f"{pre_chunk.get('title')}和next父标题相同,可以继续检查合并!")
                # parent_title\n 内容 \n 内容
                next_content = next_chunk.get("content")
                if len(pre_content + next_content[len(next_parent_title):]) < CHUNK_MAX_SIZE:
                    logger.debug(f"{pre_chunk.get('title')}和next合并后长度小于{CHUNK_MAX_SIZE},可以合并!")
                    pre_chunk["content"] = pre_content + next_content[len(next_parent_title):]
                    pre_chunk["part"] = next_chunk.get("part")
                else:
                    logger.debug(f"{pre_chunk.get('title')}和next合并后长度大于{CHUNK_MAX_SIZE},不能合并!")
                    final_chunks.append(pre_chunk)  # 之前的进行存储
                    pre_chunk = next_chunk
            else:
                logger.debug(f"{pre_chunk.get('title')}和next父标题不相同,不能合并!")
                final_chunks.append(pre_chunk)  # 之前的进行存储
                pre_chunk = next_chunk
        else:
            logger.debug(f"{pre_chunk.get('title')}对应的内容content长度大于{CHUNK_MIN},不能合并!")
            final_chunks.append(pre_chunk)  # 之前的进行存储
            pre_chunk = next_chunk
    # 循环完毕
    # 可能1: 倒数第二个 pre   倒数第一个 next 可以合并      pre.content  <- next.content
    # 可能2: 倒数第二个 pre   倒数第一个 next 不可以合并    pre -> final_chunks   pre<-next
    final_chunks.append(pre_chunk)
    logger.info(f"进行短合并处理,处理完毕的数据长度:{len(final_chunks)}")
    return final_chunks


@step_log("refine_split_or_merge_chunks")
def refine_split_or_merge_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # [  { parent_title part -> 超长二次切割!! title file_title content 400 - 600 -  1000} ]
    logger.info(f"进行精细切割或者合并的原数据长度:{len(chunks)}")
    # 1. 循环处理每个切块! > 600触发切割
    refine_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        if len(chunk.get("content")) > CHUNK_SIZE:
            # list[dict[str,Any]]
            logger.debug(f"{chunk.get('title')}对应的切块内容长度大于600,触发二次切割!!")
            refine_chunks.extend(_too_long_split(chunk))
        else:
            logger.debug(f"{chunk.get('title')}对应的切块内容长度小于600,不触发二次切割!!")
            refine_chunks.append(chunk)
    logger.info(f"进行精细切割过长的进行二次切割,处理后长度为:{len(refine_chunks)}")
    # 2. 过短的内容进行合并处理
    refine_chunks = _too_short_merge(refine_chunks)
    logger.info(f"进行精细切割过短的进行同一个父标题的合并,处理后长度为:{len(refine_chunks)}")
    # 3. 补全属性 parent_title part
    for chunk in refine_chunks:
        if "parent_title" not in chunk:
            chunk["parent_title"] = chunk.get("title")
        if "part" not in chunk:
            chunk["part"] = 1
    return refine_chunks


#  4. 数据备份处理chunk.json
@step_log("backup_chunks")
def backup_chunks(chunks: list[dict[str, Any]], md_path_obj: Path):
    # 1. 获取目标地址 md同级 chunk.json
    chunk_json_obj: Path = md_path_obj.parent / "chunk.json"
    # 2. 向指定的文件写入字符串  ensure_ascii中文正常显示
    chunk_json_obj.write_text(data=json.dumps(chunks, indent=4, ensure_ascii=False), encoding="utf-8")
    logger.info(f"已经将chunks备份到:{str(chunk_json_obj)}")


@step_log("split_document")
def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    文档切分服务：
    1. 按标题层级做一级粗切
    2. 对超长文本做二次细切
    3. 构造 chunks 列表
    4. 数据备份处理chunk.json
    4. 回写 chunks
    """
    md_content, file_title, md_path_obj = validate_data_and_content(state)
    chunks: list[dict[str, Any]] = split_document_by_title(md_content, file_title)
    refine_chunks: list[dict[str, Any]] = refine_split_or_merge_chunks(chunks)
    backup_chunks(refine_chunks, md_path_obj)
    # 更新state
    state['chunks'] = refine_chunks
    return state


"""

[
    {
        "title": "前言",
        "parent_title": "",
        "file_title": "RAG学习",
        "part": 1,
        "content": "这是内部使用的 RAG 学习材料。"
    },

    {
        "title": "向量检索",
        "parent_title": "RAG > 检索",
        "file_title": "RAG学习",
        "part": 1,
        "content": "
### 向量检索

向量检索通过 Embedding 找到语义相似文本。
"
    },

    {
        "title": "BM25",
        "parent_title": "RAG > 检索",
        "file_title": "RAG学习",
        "part": 1,
        "content": "
### BM25

BM25 根据关键词统计进行检索。
"
    },

    {
        "title": "Agent",
        "parent_title": "",
        "file_title": "RAG学习",
        "part": 1,
        "content": "
# Agent

Agent 可以调用工具。
"
    }
]


原 Chunk
{
 title
 file_title
 content
}
       ↓
发现太长
       ↓
从 content 中去掉 title
       ↓
clean_content
       ↓
RecursiveCharacterTextSplitter
       ↓
按：
段落 → 换行 → 句号 → 分号 → 逗号
寻找自然边界
       ↓
正文1   正文2   正文3
  ↓       ↓       ↓
重新加上原始 title
  ↓       ↓       ↓
Chunk1  Chunk2  Chunk3
       ↓
补：
parent_title
part
file_title
       ↓
返回 sub_chunks
"""


"""

Chunk
 ↓
判断类型

├─ 普通文本
│    ↓
│  600左右
│  >1000继续切
│
├─ 表格
│    ↓
│  优先完整保留
│  允许适度超过1000
│
├─ 代码块
│    ↓
│  优先完整保留
│
└─ 其他特殊结构
     ↓
   对应特殊规则"""