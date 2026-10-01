# 测试导入流程执行过程

# 单个 -> node -> state
# 整体 -> 图 -> state
# graph  invoke -> 日志 /  [ stream ]
# state  -> graph - invoke -> 日志 -> milvus | output可视化文件
# 期待没有bug amen

# state | graph_app
from app.process.import_.agent.state import create_default_state,ImportGraphState
from app.process.import_.agent.main_graph import import_graph_app
from app.shared.runtime.logger import logger,PROJECT_ROOT
from pathlib import Path
import uuid
import json

local_file_path_obj:Path =PROJECT_ROOT / "doc" / "hak180使用说明书.pdf"
local_dir_obj:Path = PROJECT_ROOT / "output"

# 1.创建state
state:ImportGraphState = create_default_state(
    # local_file_path:str
    local_file_path = str(local_file_path_obj),
    local_dir = str(local_dir_obj),
    # task_id:str
    task_id = str(uuid.uuid4())
)
logger.info(f"开始流程测试,测试数据:{json.dumps(state,indent=4,ensure_ascii=False)}")
# 2.执行图对象
try:
    # 开启
    result: ImportGraphState = import_graph_app.invoke(state)
    # 完成
except Exception as e:
    # 失败
    print(e)

# 3.输出最终结果
logger.info(f"测试流程结束,结果数据:{json.dumps(result,indent=4,ensure_ascii=False)}")
