import json

from app.shared.runtime.logger import logger
from app.process.import_.agent.state import ImportGraphState, create_default_state
from app.process.import_.agent.main_graph import import_graph_app

# # 动态测试
# state = create_default_state(task_id='007', local_file_path='./xxx.md')
# logger.info(f'开始进行测试：输入的数据：{json.dumps(state, indent=4, ensure_ascii=False)}')
# result = import_graph_app.invoke(state)
# logger.info(f'测试结束：输出的数据：{json.dumps(result, indent=4, ensure_ascii=False)}')

# 静态测试 不关注运行的参数，单出的输出图的节点 执行全部可能
import_graph_app.get_graph().print_ascii()
