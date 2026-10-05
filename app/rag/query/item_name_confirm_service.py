from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pymilvus import AnnSearchRequest

from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import HYBRID_SEARCH_LIMIT_NUMBER, HYBRID_SEARCH_RANKER_WEIGHTS, ITEM_NAME_CONFIRM_THRESHOLD, \
    ITEM_NAME_CANDIDATE_THRESHOLD, ITEM_NAME_OPTIONS_TOPK
from app.shared.clients import save_chat_message
from app.shared.runtime.logger import logger
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import step_log

from app.shared.clients.mongo_history_utils import get_recent_messages
from app.infra.llm.providers import llm_provider


# 1. 获取并校验参数
@step_log("get_data_and_validate")
def get_data_and_validate(state) -> tuple[str, str]:
    # 1.1获取请求参数
    session_id = state.get("session_id")
    original_query = state.get("original_query")
    # 1.2非空校验
    if not session_id or not original_query:
        logger.error(f'session_id 或者 original_query为空！业务无法继续，提前终止！')
        raise ValueError(f'session_id 或者 original_query为空！业务无法继续，提前终止！')
    # 1.3不为空返回结果
    return session_id, original_query


# 2. 先查询有效的历史聊天记录
@step_log("get_valid_history_messages")
def get_valid_history_messages(session_id) -> list[dict]:
    # 2.1 近10条聊天记录,调用history_utils查询对应session_id和limit聊天数据 -> find({session_id:session_id}).order({ts:-1}).limit(10)
    history_messages: list[dict] = get_recent_messages(session_id, limit=10)
    # 2.2 清洗数据，获取有效的聊天记录！item_names不为空！
    valid_history_messages: list[dict] = [history_message for history_message in history_messages if
                                          history_message.get('item_name')]
    # 2.3 返回有效的聊天列表
    return valid_history_messages


# 3. 调用模型使用聊天记录+原始问题提取item_names(模型提取)和重写问题
# 有效聊天记录+原始问题 -> 提示词 -> 模型 -> item_names:[] rewritten_query   模型返回json: 1.提示词说明要求 2.提示词提供示例 3.模型设置json参数 4.返回json处理和校验
@step_log("call_model_extract_item_names_and_rewritten_query")
def call_model_extract_item_names_and_rewritten_query(original_query: str, valid_history_messages: list[dict]) -> dict[
    str, Any]:
    # 3.1 准备大语言模型客户端 infra.llm_model(模型的名字,json_mode=True)
    llm_model_json = llm_provider.chat(json_mode=True)
    # 3.2 加载和封装提示词消息
    history_text: str | None = None
    if valid_history_messages:
        # role = user  text = 原始问题  rewritten_query = 重写的问题  item_names = [1,2,3] image_urls = []
        # 本次为提问,原始问题为:xx,上一次问题重写后:xxx,问题管理的item_name:x,x,x
        # role = assistant  text = 模型的回答  rewritten_query = 重写的问题  item_names = [1,2,3] image_urls
        # 本次关于: 重写问题 的回答为:xxx,关联的实体为:x,x,x
        # 提取item_names / 重写问题
        for message in valid_history_messages:
            if message.get('role') == 'user':
                history_text += f"本次是提问记录,原始问题为:{message.get('text')},问题重写后:{message.get('rewritten_query')},问题关联的item_name:{message.get('item_names')}\n"
            else:
                history_text += f"本次是回答记录,关于问题:{message.get('rewritten_query')}的回答! 回答内容为:{message.get('text')[:100]}....,关联的item_name:{message.get('item_names')}\n"
    else:
        history_text = '无有效的聊天记录！'
    prompt_text: str = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=original_query)
    messages = [HumanMessage(content=prompt_text)]
    # 3.3 定义json模型调用链  chains = 模型 | JSONOutputParser()
    chains = llm_model_json | JsonOutputParser()
    # 3.4 调用获取返回字典dict
    result_json_dict: dict = chains.invoke(messages)
    # 3.5 进行返回数据校验 有item_names属性 -> []  有rewritten_query -> 原始问题
    if "item_names" not in result_json_dict:
        logger.warning(f"{original_query}没有提取到item_names!")
        result_json_dict['item_names'] = []
    if "rewritten_query" not in result_json_dict:
        logger.warning(f"{original_query}没有完成问题重写!")
        result_json_dict['rewritten_query'] = original_query
        # 3.6 返回即可dict
    return result_json_dict


# if 返回的字典有item_names[模型识别到]且不为空:
# 4. 根据模型提取的item_names进行向量库混合检索
@step_log("query_milvus_item_names")
def query_milvus_item_names(item_names: list[str]) -> dict[str, list[dict[str, Any]]]:
    # 问题: 华为手机和苹果手机哪个好用? ->  模型 -> item_names -> [华为手机,苹果手机] -> milvus -> 混合检索
    #       -> 华为手机 -> 混合检索 -> [[{id:1,distance:0.6,entity:{item_name:华为p60手机}},{id:1,distance:0.6,entity:{item_name:华为p60手机}}...]]
    #       {华为手机:[ {item_name:"华为p60手机",distance:0.6} , {item_name:"华为p60手机",distance:0.6}, {item_name:"华为p60手机",distance:0.6}]}
    #       -> 苹果手机 -> 混合检索 -> [[{id:1,distance:0.6,entity:{item_name:苹果p60手机}},{id:1,distance:0.6,entity:{item_name:华为p60手机}}...]]
    #       {苹果手机:[ {item_name:"苹果p60手机",distance:0.6} , {item_name:"华为p60手机",distance:0.6}, {item_name:"华为p60手机",distance:0.6}]}

    # “模型先猜主体 → Milvus 再把主体对齐到知识库里的标准名称。”
    # {"dense":[[],[]],"sparse":[{},{}]}
    milvus_item_name_dict: dict[str, list[dict[str, Any]]] = {}
    # 4.1 处理模型提供的item_names
    result = llm_provider.generate_embeddings(item_names)
    for index in range(len(item_names)):
        # 4.2 将每个item_name生成稠密和稀疏向量
        dense_vector = result['dense'][index]
        sparse_vector = result['sparse'][index]
        # 4.3 根据稠密和稀疏向量创建两个request
        requests: list[AnnSearchRequest] = milvus_gateway.create_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            limit=HYBRID_SEARCH_LIMIT_NUMBER * 2
        )
        # 4.4 进行混合检索
        response = milvus_gateway.hybrid_search(
            collection_name=milvus_gateway.item_name_collection_name,
            reqs=requests,
            ranker_weights=HYBRID_SEARCH_RANKER_WEIGHTS,
            norm_score=True,
            limit=HYBRID_SEARCH_LIMIT_NUMBER
        )
        # response = [  [ {id:主键,distance: 分 , entity:{item_name:名字} }  ->  {item_name: , score: distance}] ]
        # 4.5 将检索的结果封装成 [{item_name:x,score:x},{},{} -> limit = 5]
        current_item_name_result: list[dict[str, Any]] = [{'item_name': item.get('entity', {}).get('item_name'),
                                                           'score': item.get('distance', 0.0)} for item in response[0]]
        # 4.6 最终 [{},{},{}] -> {item_name: [{},{},{}]}
        milvus_item_name_dict[item_names[index]] = current_item_name_result
        logger.debug(f"{item_names[index]},检索向量数据库的结果为:{current_item_name_result}")
    # 4.7 跳出循环返回结果
    return milvus_item_name_dict


"""
item_name = "华为手机"
      ↓
BGE-M3
      ↓
Dense + Sparse
      ↓
两个 AnnSearchRequest
      ↓
hybrid_search
      ↓
response

[
  [
    华为P60   score 0.93,
    华为Mate60 score 0.86,
    华为P50   score 0.72
  ]
]
      ↓
response[0]
      ↓
[
  华为P60,
  华为Mate60,
  华为P50
]
"""


# 5. 置信度处理(判断分)分值区分,我们获取 确定的列表 / 可选的列表 / 没有
#            高于多少分,可以确定      -> 确定阈值   ->   1 [最高分]
#            低于确定的阈值,高于多少分 -> 可选的阈值 ->   2~3
#            低于的可选              -> 不确定    ->   0
#            {
#              item_name:[{item_name:x,distance:分},{}]  -> 确定 / 可选 / 不确定
#              item_name:[{item_name:x,distance:分},{}]  -> 确定 / 可选 / 不确定
#              item_name:[{item_name:x,distance:分},{}]  -> 确定 / 可选 / 不确定
#            }
#            确定的列表: [1,1,1,1]         -> item_names -> state
#            可选的列表: [1,2,3,1,2,3]     -> answer     -> out..
#            以上都为空:                   -> answer     -> out..
@step_log("select_item_names")
def select_item_names(milvus_item_name_dict: dict[str, list[dict[str, Any]]]) -> dict[str, list]:
    # 5.1 定义两个列表 confirmed_list | optional_list
    confirmed_list: list[str] = []
    optional_list: list[str] = []
    # 5.2 定义阈值 ... 三个
    # 主体名称确认阈值：高于该分数 → 直接确认 [0.75]
    # ITEM_NAME_CONFIRM_THRESHOLD = 0.65
    # # 主体名称候选阈值：介于两者之间 → 让用户选择
    # ITEM_NAME_CANDIDATE_THRESHOLD = 0.50
    # # 给用户选择时，最多展示几个候选
    # ITEM_NAME_OPTIONS_TOPK = 2
    """
    1. **离线标定**：用业务真实样本，统计分数分布，划分三档：高置信、待校验、丢弃，预留浮动余量，不拿理想满分当阈值。
          HAK 180烫金机 -> 高置信 180烫金机  烫金机  HAK 烫金机   HAK 180  -> 字符串差别对比
                       -> 待校验 780打印机 xx   
    2. **线上粗分层**：hybrid 返回结果按标定阈值分成 3 组。
    3. 强制二次校验
       - 高置信：直接入上下文，必须过 LLM 校验兜底
       - 待校验：全部走 Reranker 精排，再交 LLM
       - 丢弃：直接扔掉
    4. **迭代维护**：改权重 / 模型 / 语料，重新标定阈值；线上观察误召、漏召情况调整。
    > 约束：只适用于 WeightedRanker；RRF 不能用分数过滤；调大两路 ann 的 limit 减少分数波动；hybrid 分数只做粗筛，不做最终判定。
    """
    # 5.3 循环结果字典  for item_name,查询的列表  in .items()
    for model_item_name, milvus_list in milvus_item_name_dict.items():
        # 5.4 根据阈值选择确定的列表 -> 取第一个放到confirmed_list
        # item => {item_name:x,分数}
        confirm_milvus_list: list[str] = [item.get('item_name') for item in milvus_list if
                                          item.get('score') >= ITEM_NAME_CONFIRM_THRESHOLD]
        if confirm_milvus_list:
            # 有确定的
            logger.debug(f"{model_item_name}有确定的高分查询item_name:{confirm_milvus_list[0]}")
            confirmed_list.append(confirm_milvus_list[0])
            continue

        # 5.5 根据阈值选择可选的列表 -> 取topk(2~3)放到optional_list
        optional_milvus_list: list[str] = [item.get('item_name') for item in milvus_list if
                                           item.get('score') >= ITEM_NAME_CANDIDATE_THRESHOLD and item.get(
                                               'score') < ITEM_NAME_CONFIRM_THRESHOLD]

        if optional_milvus_list:
            # 没有确定，但是有可选的
            logger.debug(
                f"{model_item_name}没有有确定的高分查询item_name,但是有可选的:{optional_milvus_list[:ITEM_NAME_OPTIONS_TOPK]}")
            optional_list.extend(optional_milvus_list[:ITEM_NAME_OPTIONS_TOPK])
            continue
    # 5.6 循环结束 -> 返回dict confirmed_list | optional_list
    return {
        'confirmed_list': confirmed_list,
        'optional_list': optional_list
    }


"""
LLM提取 item_name
        ↓
Milvus Dense + Sparse
        ↓
WeightedRanker
        ↓
得到 TopK + score
        ↓
       分档
        ↓
┌─────────────┬─────────────┬─────────────┐
↓             ↓             ↓
高置信         中等           低置信
>=0.65       0.50~0.65       <0.50
↓             ↓             ↓
confirmed     optional       丢弃
候选           Top2~3
↓             ↓
LLM兜底        Reranker
确认           ↓
               LLM/用户确认

最后再强调一点：
0.65 / 0.50 不是通用标准答案，只是示例阈值。真正项目里应该拿你自己的商品数据跑一批正确/错误样本，再决定阈值。

所以这段代码的本质就是：
Milvus 搜索负责“给候选打分”，阈值负责“粗分级”，Reranker + LLM 负责“最终别认错人”。
"""


# 6. 结果处理 根据列表 修改state item_names + rewritten_query (确定的列表) | answer 可选 | answer 没有
@step_log("modify_state_by_dict")
def modify_state_by_dict(select_item_names_dict: dict[str, list[str]], rewritten_query: str, state: QueryGraphState):
    # 思路: 有确认,就算成功! 就算提问 2个 一个确认 | 一个不确定可选.. 也是成功!  优先判断 确定列表
    confirmed_list: list[str] = select_item_names_dict.get('confirmed_list')
    optional_list: list[str] = select_item_names_dict.get('optional_list')
    # 6.1 确定
    if confirmed_list:
        # item_names -> 确定列表
        state['item_names'] = confirmed_list
        # rewritten_query = 重写问题
        state['rewritten_query'] = rewritten_query
        logger.info(
            f"confirmed_list确认的列表有数据,直接确定item_names:{confirmed_list},rewritten_query:{rewritten_query}")
        return
    # 6.2 不确定
    if optional_list:
        # answer = 您 ... xx
        state["answer"] = f"本次提问没有明确关联主体,查出可能为:{optional_list},请您确认,再次提问!"
        logger.info(
            f"optional_list可选的列表有数据!本次提问没有明确关联主体,查出可能为:{optional_list},请您确认,再次提问!")
        return
    # 6.3 完全不确定
    state["answer"] = f"本次提问没有任何可选关联主体,请您确认,再次提问!"
    logger.info(f"confirmed_list/optional_list列表没有数据!本次提问没有任何可选关联主体,请您确认,再次提问!")


@step_log("confirm_item_name")
def confirm_item_name(state: QueryGraphState) -> QueryGraphState:
    # 1. 获取并校验参数
    session_id, original_query = get_data_and_validate(state)
    # 2. 获取有效的聊天记录
    valid_history_messages = get_valid_history_messages(session_id)
    # 3. 调用模型提取item_names和重写问题
    result_json_dict = call_model_extract_item_names_and_rewritten_query(original_query, valid_history_messages)
    # 4. 判断result_json_dict item_names不为空
    select_item_names_dict = {}
    if result_json_dict.get("item_names"):
        # 5.调用数据库查询item_names关联的数据库主体和分数 {item_name:[{item_name:x,score:x}]}
        milvus_item_name_dict = query_milvus_item_names(result_json_dict.get("item_names"))
        # 6.处理结果,置信度处理得到确认和可选的列表
        select_item_names_dict = select_item_names(milvus_item_name_dict)
    # 7.根据确认和可选的列表修改state
    modify_state_by_dict(select_item_names_dict, result_json_dict.get("rewritten_query"), state)
    # 8. 记录聊天记录
    save_chat_message(
        session_id=state.get("session_id"),
        role='user',
        text=state.get("original_query"),
        rewritten_query=f'{state.get("rewritten_query")}',
        item_names=state.get("item_names"),
        image_urls=[]
    )
    return state


"""
用户原始问题
    ↓
历史聊天 + original_query
    ↓
LLM
    ↓
提取候选 item_names
+ rewritten_query
    ↓
例如：
["华为手机", "苹果手机"]
    ↓
逐个 item_name
    ↓
BGE-M3
    ↓
Dense + Sparse
    ↓
AnnSearchRequest × 2
    ↓
Milvus hybrid_search
    ↓
每个 item_name 得到 TopK 候选
    ↓
整理成：

{
  原始item_name: [候选...]
}

    ↓
select_item_names()
    ↓
按阈值判断
    ↓
┌────────────┬────────────┬────────────┐
↓            ↓            ↓
确定          可选          不确定
↓            ↓            ↓
Top1         Top2~3        0个
↓            ↓            ↓
item_names   answer        answer
↓            ↓            ↓
继续检索      让用户确认     让用户重说
"""
