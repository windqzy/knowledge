from langgraph.graph import StateGraph, END

from app.process.query.agent.nodes.node_answer_output import node_answer_output
from app.process.query.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.process.query.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.process.query.agent.state import QueryGraphState, query_graph_default_state
from app.shared.runtime.logger import logger

# 1.创建查询图的编译对象 builder
query_graph_builder = StateGraph(state_schema=QueryGraphState)
# 2.添加节点
query_graph_builder.add_node(node_item_name_confirm)
query_graph_builder.add_node(node_search_embedding)
query_graph_builder.add_node(node_search_embedding_hyde)
query_graph_builder.add_node(node_web_search_mcp)
query_graph_builder.add_node(node_rrf)
query_graph_builder.add_node(node_rerank)
query_graph_builder.add_node(node_answer_output)
# 3.添加边(起始边/条件边/静态边)
query_graph_builder.set_entry_point('node_item_name_confirm')


# 条件边
def after_node_item_name_confirm(state: QueryGraphState):
    answer = state.get('answer')
    if answer:
        logger.info(f'node_item_name_confirm没有确认item_name,所以跳转node_answer_output，直接输出answer')
        return 'node_answer_output'
    else:
        logger.info(
            f'node_item_name_confirm确认item_name：{state.get('item_names')},所以跳转node_item_name_confirm，正常进行多路召回')
        return 'node_search_embedding', 'node_search_embedding_hyde', 'node_web_search_mcp'


query_graph_builder.add_conditional_edges("node_item_name_confirm",
                                          after_node_item_name_confirm,
{
    'node_answer_output': 'node_answer_output',
    'node_search_embedding': 'node_search_embedding',
    'node_search_embedding_hyde': 'node_search_embedding_hyde',
    'node_web_search_mcp': 'node_web_search_mcp',
})
"""
node_item_name_confirm
        ↓
after_node_item_name_confirm(state)
        ↓
看 answer

    ┌──────────────┴──────────────┐
    ↓                             ↓
answer有值                     answer没值
    ↓                             ↓
node_answer_output          三路召回
                           ├─ embedding
                           ├─ hyde
                           └─ web
合并冲突怎么解决呢？？
#并发的地方不能有相同的key，有就报错
"""

query_graph_builder.add_edge('node_search_embedding', 'node_rrf')
query_graph_builder.add_edge('node_search_embedding_hyde', 'node_rrf')
query_graph_builder.add_edge('node_web_search_mcp', 'node_rrf')
query_graph_builder.add_edge('node_rrf', 'node_rerank')
query_graph_builder.add_edge('node_rerank', 'node_answer_output')
query_graph_builder.add_edge('node_answer_output', END)

# 4.编译获取查询图对象
query_graph_app = query_graph_builder.compile()
