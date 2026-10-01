from app.shared.runtime.logger import logger
from app.process.import_.agent.state import create_default_state,get_default_state

#1.创建一个修改local_file_path/task_id state
state1 = create_default_state(task_id = '007',local_file_path='./中文.pdf')
logger.info(state1)

import json
#json.dumps() json.dump()
# python对象转成json s表示json字符串 不带s表示转成文件中的
#json.loads() json.load()
# json字符串转成python的字典对象

state2 = create_default_state(local_file_path='./xxx.pdf')
logger.info(f'state2 = \n{json.dumps(state2,indent=4,ensure_ascii=False)}')