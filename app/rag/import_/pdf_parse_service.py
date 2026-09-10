import time
from pathlib import Path

import requests
import shutil

from scipy.signal import step

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import PARSE_PDF_OUTPUT_DIR, MINERU_MODEL_VERSION, MINERU_DOWNLOAD_TIMEOUT_SECONDS, \
    MINERU_POLL_INTERVAL_SECONDS, MINERU_POLL_TIMEOUT_SECONDS
from app.shared.runtime.logger import logger, PROJECT_ROOT, step_log
from app.infra.config.providers import infra_config

@step_log('validate_data_and_paths')
def validate_data_and_paths(state: ImportGraphState) -> tuple[Path]:
    # 获取参数
    pdf_path: str = state.get('pdf_path')
    local_dir: str = state.get('local_dir')

    # 非空校验
    if not pdf_path:
        logger.error(f'pdf_path为空，没有文件可以解析，业务终止。')
        raise ValueError(f'pdf_path为空，没有文件可以解析，业务终止。')
    if not local_dir:
        local_dir: Path = PROJECT_ROOT / PARSE_PDF_OUTPUT_DIR
        logger.warning(f'local_dir为空，为了业务继续进行，给予默认值:{str(local_dir)}')
    # 转为path
    pdf_path_obj: Path = Path(pdf_path)
    local_dir_obj: Path = Path(local_dir)
    # 存在性校验
    if not pdf_path_obj.is_file():
        logger.error(f'{str(local_dir_obj)}这里并没有文件，无法继续业务,业务终止。')
        raise FileNotFoundError(f'{str(local_dir_obj)}这里并没有文件夹，无法继续业务,业务终止。')
    if not local_dir_obj.is_dir():
        logger.warning(f'{str(local_dir_obj)}这里并没有文件夹，无法继续业务,开始创建文件夹。')
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    # 返回结果
    return pdf_path_obj, local_dir_obj

@step_log('upload_pdf_and_poll')
def upload_pdf_and_poll(pdf_path_obj: Path) -> ImportGraphState:
    # 1.向mineru申请一个上传的地址
    token = infra_config.mineru.api_key
    url = f'{infra_config.mineru.base_url}/file-urls/batch'
    headers = {'Content-Type': 'application/json',
               'Authorization': f'Bearer {token}'}
    data = {
        'files': [
            {'name': f'{pdf_path_obj.name}'}
        ],
        'model_version': MINERU_MODEL_VERSION
    }
    response = requests.post(url, headers=headers, json=data, timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)
    status_code = response.status_code
    if status_code != 200:
        logger.error(f'向mineru申请文件解析地址，网络状态错误：{status_code},业务无法继续，提前终止。')
        return RuntimeError(f'向mineru申请文件解析地址，网络状态错误：{status_code},业务无法继续，提前终止。')
    # 响应体的json字符串
    response_json_dict: dict = response.json()
    code = response_json_dict.get('code', -1)
    if code != 0:
        logger.error(f'向mineru申请文件解析地址，业务状态错误：{code},业务无法继续，提前终止。')
        return RuntimeError(f'向mineru申请文件解析地址，业务状态错误：{code},业务无法继续，提前终止。')
    batch_id: str = response_json_dict.get('data', {}).get('batch_id')
    file_urls: list[str] = response_json_dict.get('data', {}).get('file_urls', [])
    # 非空校验
    if not batch_id:
        logger.error(f'向mineru申请文件解析地址，batch_id为空,业务无法继续，提前终止。')
        return RuntimeError(f'向mineru申请文件解析地址，batch_id为空,业务无法继续，提前终止。')
    if not file_urls:
        logger.error(f'向mineru申请文件解析地址，file_urls为空,业务无法继续，提前终止。')
        return RuntimeError(f'向mineru申请文件解析地址，file_urls为空,业务无法继续，提前终止。')
    file_url: str = file_urls[0]
    logger.info(f'成功向mineru申请到文件解析地址，batch_id:{batch_id},file_url:{file_url}')

    # 2.向指定地址发起请求上传pdf文件
    with requests.Session() as session:
        data = pdf_path_obj.read_bytes()
        # 不要自动信任/读取当前操作系统环境里的某些网络配置。
        session.trust_env = False
        upload_response = session.put(url=file_url, data=data)
        upload_status_code = upload_response.status_code
        if upload_status_code != 200:
            logger.error(f'向指定地址：{file_url}上传文件失败，状态码为：{upload_status_code},业务无法继续，提前终止。')
            return RuntimeError(
                f'向指定地址：{file_url}上传文件失败，状态码为：{upload_status_code},业务无法继续，提前终止。')
    logger.info(f'向指定地址：{file_url}上传文件成功！')

    # 3.使用batch_id轮询获取解析结果
    # 请求地址 间隔时间 最大等待时间
    url = f'{infra_config.mineru.base_url}/extract-results/batch/{batch_id}'
    interval_time = MINERU_POLL_INTERVAL_SECONDS
    timeout_time = MINERU_POLL_TIMEOUT_SECONDS
    current_time = time.time()
    # 轮询就是一个死循环！满足条件跳出。
    while True:
        # 3.1有没有超出等待时间
        if time.time() - current_time > timeout_time:
            logger.error(f'轮询获取解析结果超时！提前终止业务。')
            raise TimeoutError(f'轮询获取解析结果超时！提前终止业务。')
        # 3.2没有超时就发起请求
        poll_response = requests.get(url=url, headers=headers)
        # 3.3 判断网络状态码
        if poll_response.status_code != 200:
            if 500 <= poll_response.status_code <= 600:
                logger.warning(f'轮询获取解析结果，状态码为：{poll_response.status_code},再次尝试')
                time.sleep(interval_time)
                # asyncio.sleep(interval_time)
                continue
            else:
                logger.error(f'轮询获取解析结果，状态码为：{poll_response.status_code},业务无法继续，提前终止。')
                raise RuntimeError(f'轮询获取解析结果，状态码为：{poll_response.status_code},业务无法继续，提前终止。')

        # 3.4判断业务状态码
        poll_response_dict: dict = poll_response.json()
        poll_code = poll_response_dict.get('code', -1)
        if poll_code != 0:
            logger.error(f'向mineru获取文件解析结果，业务状态错误：{code},业务无法继续，提前终止。')
            return RuntimeError(f'向mineru获取文件解析结果，业务状态错误：{code},业务无法继续，提前终止。')
        # 3.5获取解析结果
        poll_result_dict: dict = poll_response_dict.get('data', {}).get('extract_result', [])[0]
        poll_state: str = poll_result_dict.get('state')
        if poll_state == 'done':
            full_zip_url = poll_result_dict.get('full_zip_url')
            logger.info(f'向mineru获取文件解析结果，业务状态为：成功。解析好的文件下载地址为：{full_zip_url}')
            return full_zip_url
        elif poll_state == 'failed':
            logger.error(f'向mineru获取文件解析结果，业务状态为：失败。')
            raise RuntimeError(f'向mineru获取文件解析结果，业务状态为：失败。')
        else:
            # 没失败 也没解析完毕
            logger.info(f'向mineru获取文件解析结果，业务状态为：{poll_state},再等待一会儿。')
            time.sleep(interval_time)
            continue

@step_log('download_zip_and_extract_md')
def download_zip_and_extract_md(zip_url: str, local_dir_obj: Path, file_name: str) -> Path:
    # 1.下载文件到output中，名称为 文件名.zip
    response = requests.get(url=zip_url, timeout=MINERU_DOWNLOAD_TIMEOUT_SECONDS)
    status_code = response.status_code
    if status_code != 200:
        logger.error(f"mineru返回的压缩包文件地址：{zip_url},响应状态错误：{status_code},业务无法继续")
        raise RuntimeError(f"mineru返回的压缩包文件地址：{zip_url},响应状态错误：{status_code},业务无法继续")
    # 下载到output文件夹中
    zip_file_obj: Path = local_dir_obj / f'{file_name}.zip'
    zip_file_obj.write_bytes(data=response.content)

    # 2.价差是否存在解压的文件夹(/output/文件名)
    # 检查文件夹是否已经存在
    zip_file_dir_obj: Path = local_dir_obj / file_name
    if zip_file_dir_obj.is_dir():
        # 递归清空文件夹中的内容
        shutil.rmtree(zip_file_dir_obj)
    zip_file_dir_obj.mkdir(parents=True, exist_ok=True)
    # 解压
    shutil.unpack_archive(filename=zip_file_obj, extract_dir=zip_file_dir_obj)
    # 3.检查文件夹中是否存在md文件 full.md->file_name.md
    md_file_list: list[Path] = list(zip_file_dir_obj.rglob('*.md'))
    if not md_file_list:
        logger.error(f'mineru返回的压缩包文件地址：{zip_url},解压后里面没有md文件,业务无法继续')
        raise FileNotFoundError(f'mineru返回的压缩包文件地址：{zip_url},解压后里面没有md文件,业务无法继续')
    # 重命名:可能1:原文件名.md 可能2:full.md
    for md_file in md_file_list:
        if md_file.stem == file_name:
            logger.info(f'mineru返回的压缩包文件地址：{zip_url},md文件地址为:{md_file}')
            return md_file
    for md_file in md_file_list:
        if md_file.stem == 'full':
            # 重命名并返回即可  rename理解成“移动+重命名”，不是复制
            # md_file_obj: Path = md_file.rename(f'{file_name}.md')
            target_md_path = md_file.with_name(f'{file_name}.md')
            md_file_obj = md_file.rename(target_md_path)
            logger.info(f'mineru返回的压缩包文件地址：{zip_url},md文件地址为:{md_file_obj}')
            return md_file_obj
    logger.info(f'mineru返回的压缩包文件地址：{zip_url},解压后文件名不叫full或者是原文件名，请根据官网确认后再解析')
    raise FileNotFoundError(
        f'mineru返回的压缩包文件地址：{zip_url},解压后文件名不叫full或者是原文件名，请根据官网确认后再解析')

@step_log('parse_pdf_to_markdown')
def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:
    """
    PDF 解析服务：
    1. 调用 MinerU
    2. 下载并解压解析结果
    3. 获取 Markdown 路径和正文内容
    4. 回写 md_path / md_content / local_dir
    """
    # 1.生成pdf_path 和local_dir 为pdf解析为md做准备
    pdf_path_obj, local_dir_obj = validate_data_and_paths(state)
    # 2.mineru申请上传地址 并上传 后轮询 获得解析结果
    full_zip_url: str = upload_pdf_and_poll(pdf_path_obj)
    # 3.下载并解压zip 并重命名md文件
    md_path_obj: Path = download_zip_and_extract_md(zip_url=full_zip_url, local_dir_obj=local_dir_obj,
                                                    file_name=pdf_path_obj.stem)
    # 4.更新state md_path属性
    state['md_path'] = str(md_path_obj)
    return state


"""
                ① POST
你的程序 ─────────────────→ MinerU API
          “给我一个上传地址”

你的程序 ←───────────────── MinerU API
          batch_id
          file_url


                ② PUT
你的程序 ─────────────────→ file_url
                            对象存储
          “这是我的PDF文件”


                ③ GET
你的程序 ─────────────────→ MinerU API
          “解析完了吗？”

你的程序 ←───────────────── MinerU API
          processing

                ↓
              轮询

你的程序 ─────────────────→ MinerU API
          “解析完了吗？”

你的程序 ←───────────────── MinerU API
          done
          full_zip_url
          

State
│
│ pdf_path: str
↓
Service
│
│ Path(pdf_pat
↓
Path对象
│
├── is_file()
├── read_bytes()
├── stem
├── parent
├── rename()
└── ...
│
↓
处理完成
│
│ str(md_path)
↓
State
│
│ md_path: str
└── md_content: str
"""
