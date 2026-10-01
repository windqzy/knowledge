
from app.shared.runtime.logger import logger
from app.process.query.agent.state import QueryGraphState,create_query_default_state
from app.process.query.agent.main_graph import query_graph_app
import json

#1.创建图需要的state参数
state: QueryGraphState = create_query_default_state(
    session_id='session_id007',
    original_query = '烫金机是用来干什么的？'
)
logger.info(f"开始测试查询图流程：传入参数为：\n{json.dumps(state, indent=4,ensure_ascii=False)}")
#2.调用图对象
result = query_graph_app.invoke(state)
logger.info(f"测试结束查询图流程：查询结果为：\n{json.dumps(result, indent=4,ensure_ascii=False)}")

#3.静态测试
query_graph_app.get_graph().print_ascii()