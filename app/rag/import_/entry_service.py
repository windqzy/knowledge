from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger, step_log
from pathlib import Path

@step_log("resolve_input_file")
def resolve_input_file(state: ImportGraphState) -> ImportGraphState:
    """
    入口识别服务：
    1. 校验 local_file_path
    2. 识别文件类型（PDF / Markdown）
    3. 回写 is_pdf_read_enabled / is_md_read_enabled
    4. 回写 pdf_path / md_path / file_title
    """

    # 1.state获取参数 local_file_path
    local_file_path = state.get('local_file_path')
    # 2.非空校验
    if not local_file_path:
        logger.error(f'local_filt_path没有赋值，没有文件可以解析，提前终止！')
        # 为空
        raise ValueError(f'local_filt_path没有赋值，没有文件可以解析，提前终止！')
    # 3.判断文件类型 .md .pdf
    if local_file_path.lower().endswith('.md'):  # md
        state['is_md_read_enabled'] = True
        state['md_path'] = local_file_path
        state['is_pdf_read_enabled'] = False
        state['pdf_path'] = None
    elif local_file_path.lower().endswith('.pdf'):  # pdf
        state['is_pdf_read_enabled'] = True
        state['pdf_path'] = local_file_path
        state['is_md_read_enabled'] = False
        state['md_path'] = None
    else:
        state['is_pdf_read_enabled'] = False
        state['pdf_path'] = None
        state['is_md_read_enabled'] = False
        state['md_path'] = None
    # 4.获取到local_file_path 对应的文件名 给file_title赋值
    local_file_path_obj: Path = Path(local_file_path)
    # stem 文件名不带后缀
    # name 文件名带后缀
    # suffix 后缀
    file_title = local_file_path_obj.stem
    state['file_title'] = file_title
    return state


