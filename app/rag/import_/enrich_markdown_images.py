import re
from re import Match
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio import Minio
from minio.datatypes import Object
from minio.deleteobjects import DeleteObject

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_ import config
from app.rag.import_.config import SUB_MD_CONTENT_CONTEXT_LENGTH
from app.shared.runtime.logger import logger, step_log
from app.infra.llm.providers import llm_provider
from app.shared.runtime.load_prompt import load_prompt
import base64
from mimetypes import guess_type

from app.shared.utils.rate_limit_utils import apply_api_rate_limit
from app.infra.object_storage.minio_gateway import minio_gateway

@step_log("validate_data_and_paths")
def validate_data_and_paths(state: ImportGraphState) -> tuple[str, str, Path]:
    # 1.1 获取请求参数md_path
    md_path: str = state.get('md_path')
    # 1.2非空校验
    if not md_path:
        logger.error(f"md_path为空，无法读取文档，业务无法继续，提前终止！")
        raise ValueError(f"md_path为空，无法读取文档，业务无法继续，提前终止！")
    # 1.3md_path转化成md_path_obj:Path
    md_path_obj = Path(md_path)
    # 1.4文件存在性校验
    if not md_path_obj.is_file():
        logger.error(f"{md_path}地址存在，但文件不存在，业务无法继续，提前终止！")
        raise FileNotFoundError(f"{md_path}地址存在，但文件不存在，业务无法继续，提前终止！")
    # 1.5基于md_path_obj读取文件内容 md_content
    md_content = md_path_obj.read_text(encoding="utf-8")
    state['md_content'] = md_content
    # 1.6 基于md_path_obj 获取 images_path_obj对象
    images_dir_obj: Path = md_path_obj.parent / 'images'
    # 1.7 返回三个核心参数
    return md_content, md_path_obj, images_dir_obj

@step_log("extract_image_context_info")
def extract_image_context_info(images_dir_obj: Path, md_content) -> list[tuple[str, Path, tuple[str, str]]]:
    image_context_list: list[tuple[str, Path, tuple[str, str]]] = []
    # 3.1遍历循环images_path_obj里的每个文件
    for file_obj in images_dir_obj.iterdir():
        # 3.2检查是否是图片
        if file_obj.suffix not in config.SUPPORTED_IMAGE_EXTENSIONS:
            # 不是图片 循环，源码包 工具类中logger使用debug
            logger.debug(f'本次处理的：{file_obj}不是一张图片，所 以跳过，进行下一次处理！')
            continue
        # 3.3是图片 定义图片对应的正则编译对象 re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        image_name = file_obj.name
        image_re = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        # 3.4调用正则编译对象去md_content找到匹配内容 .match[search] finditer findall
        image_match: Match = image_re.search(md_content)
        # 3.5对Match对象进行非空检查，为空-->图片没有在md_content中使用 可能是表格之类的被识别为了图片 但markdown可以正常显示出来
        if not image_match:
            logger.debug(f'本次处理的：{file_obj}是一张图片，但是没有被md_content引用，所以跳过，进行下一张图片处理！')
            continue
        # 3.6引用了 根据Match获取图片的定位信息 .start() .end() 获取上下文信息 生成对图片的描述
        start = image_match.start()
        end = image_match.end()
        # 3.7截出上文 和下文
        # config 中定义 SUB_MD_CONTENT_CONTEXT_LENGTH
        pre_context = md_content[max(0, start - SUB_MD_CONTENT_CONTEXT_LENGTH):start]
        post_context = md_content[end:min(end + SUB_MD_CONTENT_CONTEXT_LENGTH, len(md_content))]

        """
        1.SUB_MD_CONTENT_CONTEXT_LENGTH 为啥是100
        2.开头和结尾的问题 越界
        3.连续图片的问题：跳过图片继续截取
        ![]() ![]()20+80
        ![]() 10 ![]()20+70
        
        
        Markdown Parser

                ↓
        AST解析
                ↓
        
        Text Chunk
        Image Node
        Table Node  之后有时间就优化一下这里
                ↓
        Multimodal Embedding
                ↓
        Vector Database
                ↓
        Hybrid Retrieval
        """
        # 3.8拼接单个元素的原则(path.name,path,(上文，下文)) list.append()
        image_context_list.append((image_name, file_obj, (pre_context, post_context)))

    # 3.9跳出循环，打印日志，返回结果list
    logger.info(
        f'图片上下文识别结束，识别到图片的数量：{len(image_context_list)},参考示例：{'images文件夹都是非图片文件！' if len(image_context_list) == 0 else '参考示例：' + str(image_context_list[0])}')
    return image_context_list

@step_log("call_vision_summary_images")
def call_vision_summary_images(images_context_list: list[tuple[str, Path, tuple[str, str]]], file_name: str) -> dict[
    str, str]:
    # 字典
    images_summaries: dict[str, str] = {}
    # 4.1获取视觉模型对象
    # lv_model = llm_provider.vision_model(model_name=lm_config.lv_model)
    lv_model = llm_provider.vision_model(model_name='qwen2.5vl:7b')
    # 4.2循环每张图片对应的上下文信息（图片名，完成地址，(上，下文)）
    for image_name, image_path_obj, image_content in images_context_list:
        # 访问限制 不能超过模型每分钟数量 3000 60秒窗口，不能超过3000次请求
        apply_api_rate_limit(max_requests=3000, window_seconds=60)
        # 访问限制 不能超过模型每分钟数量
        # 4.3加载和凭借对应的提示词 load_prompts("image_summary",root_folder = stem,image_content(上，下文))
        summary_prompt_text: str = load_prompt("image_summary", root_folder=file_name, image_content=image_content)
        # 4.4提示词封装成Message HumanMessage(content=提示词)
        image_base64_data: str = base64.b64encode(image_path_obj.read_bytes()).decode(encoding='utf-8')
        image_mimetype: str = guess_type(image_name)[0]
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": summary_prompt_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_mimetype};base64,{image_base64_data}"
                    }
                }
            ]
        )
        # 4.5封装一个调用chains = 模型｜StrOutputParser()/JsonOutputParser
        chains = lv_model | StrOutputParser()
        # 4.6调用模型的链 chains.invoke([human_message]) -> 结果要的字符串
        image_summary = chains.invoke([message])
        # 4.7存储图片名字 和 图片summary的字典
        images_summaries[image_name] = image_summary
        logger.debug(f'本次识别的图片名：{image_name},识别的语义为：{image_summary}')
    # 4.8打印日志，返回结果
    logger.info(f'完成图片的语义识别，本次识别的数量为:{len(images_summaries)},参考数据：{images_summaries}')
    return images_summaries


# 5.掉用minio将图片传递到文件服务器 返回{图片名：网络地址}
@step_log("upload_to_minio_return_url")
def upload_to_minio_return_url(images_context_list: list[tuple[str, Path, tuple[str, str]]], file_name: str) -> dict[
    str, str]:
    images_url: dict[str, str] = {}
    # 5.1获取minio的客户端对象
    minio_client: Minio = minio_gateway.minio_client
    # 5.2先根据固定的前缀查询存不存在图片 list_objects
    select_object_list: list[Object] = minio_client.list_objects(
        bucket_name=minio_gateway.bucket_name,
        prefix=minio_gateway.image_dir[1:] + '/' + file_name,
        recursive=True
    )
    # 5.3 如果之前上传过就删除
    delete_object_list: list[DeleteObject] = [DeleteObject(obj.object_name) for obj in select_object_list]
    if delete_object_list:
        logger.info(f'{file_name}对应的文件已经存储过。数量{len(list(delete_object_list))}先删除再次上传！')
        # 5.3如果当前文件夹存在图片，先删除remove_objects
        errors = minio_client.remove_objects(minio_gateway.bucket_name,
                                             delete_object_list=delete_object_list,
                                             )
        for error in errors:
            logger.debug(f'图片处理异常:{error}')
        logger.info(f'文件的历史图片被清空！')
    # 5.4 循环上传每张图片到minio
    for image_name, image_path_obj, _ in images_context_list:
        try:
            object_name = minio_gateway.image_dir + '/' + file_name + '/' + image_name
            # 5.5 minio上传图片
            minio_client.fput_object(bucket_name=minio_gateway.bucket_name,
                                     object_name=object_name,
                                     file_path=str(image_path_obj),
                                     content_type=guess_type(image_name)[0],
                                     )

            # 5.6 为每个图片构建访问地址
            image_url = minio_gateway.build_image_url(
                stem=file_name,
                image_name=image_name
            )
            # 5.7添加到字典{图片名：图片访问地址}
            images_url[image_name] = image_url
            logger.debug(f'{image_name}已经完成上传，对应地址为：{image_url}')
        except Exception as e:
            logger.warning(f'{image_name}上传失败，跳过，继续下一张图片处理！')
            print(e)
            continue
    # 5.8返回字典
    return images_url


# 6.使用正则进行md_content内容的替换(md_content,{图片名：语义...})--->变成新的md_content
@step_log("replace_old_md_content")
def replace_old_md_content(md_content: str, images_summaries: dict[str, str], images_url: dict[str, str]) -> str:
    # 6.1 循环
    for image_name, images_summary in images_summaries.items():
        # 6.2 获取 图片名/语义/网络地址
        image_url = images_url.get(image_name)
        # 6.3 正则
        image_re = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        # 6.4正则 ![](./image_name)-->替换成![语义](图片的网络地址)
        # md_content = image_re.sub(f'![{images_summary}]({image_url})', md_content)
        # 避免特殊符号处理 报错。用lambda可以跳过特殊处理
        md_content = image_re.sub(lambda _ : f'![{images_summary}]({image_url})', md_content)
        """
        ![](images/c61a7.jpg)替换为
        ![禁止使用剪刀剪断电源线](http://minio.xxx/images/hak180/c61a7.jpg)
        """
    # 6.5循环结束
    return md_content


# 7.对新的md_content进行备份(md_content,md_path_obj) ->new_md_content地址
# 保存增强后的 Markdown
@step_log("backup_new_md_content")
def backup_new_md_content(new_md_content: str, md_path_obj: Path) -> Path:
    # 7.1获取新的md_path的目标地址 在同一个文件夹中生成一个新的文件{md_path_obj.stem}_new.md
    # 然后写入到这个位置
    new_md_path_obj: Path = md_path_obj.with_name(f'{md_path_obj.stem}_new.md')
    # 7.2将内容写到新的地址 new_md_path_obj
    new_md_path_obj.write_text(new_md_content, encoding='utf-8')
    # 7.3返回新的地址
    return new_md_path_obj

@step_log("enrich_markdown_images")
def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    """
    Markdown 图片增强服务：
    1. 扫描 Markdown 中的图片
    2. 调用多模态模型生成图片说明
    3. 上传图片到 MinIO
    4. 替换 Markdown 图片地址并回写 md_content
    """

    # 1.获取参数并校验，并生成md_path:str md_content:str images_dir_obj:Path
    md_content, md_path_obj, images_dir_obj = validate_data_and_paths(state)
    # 2.如果不是文件夹 或者 文件夹中没有文件 直接跳转到下一个节点
    if (not images_dir_obj.is_dir()) or (not list(images_dir_obj.iterdir())):
        logger.info(f"{md_path_obj}文件中没有图片，无需处理图片，直接跳转到下一个节点")
        return state
    # 3.查找images_path_obj每张图片对应的上下文信息(images_path_obj,md_content)->[(图片名.name,图片的完整地址：str/Path,(上文，下文))]
    image_context_list: list[tuple[str, Path, tuple[str, str]]] = extract_image_context_info(images_dir_obj, md_content)
    if not image_context_list:
        logger.info(
            f"{md_path_obj}：md文件，images文件中可能包含多个文件，但是没有图片被引用，无需处理图片，直接跳转到下一个节点")

    # 4.调用视觉模型进行图片请求识别[(图片名.name，图片的完整地址：str/Path，（上文，下文)，文件夹.stem]--->{图片名：图片描述...}
    images_summaries: dict[str, str] = call_vision_summary_images(image_context_list, md_path_obj.stem)
    """
    这是“{root_folder}”文件中的一张图片，图片上文部分为“{image_content[0]}”，
下文部分为“{image_content[1]}”，请用中文简要总结这张图片的内容，用于 Markdown 图片标题，控制在50字以内。
    """
    # 5.掉用minio将图片传递到文件服务器 返回{图片名：网络地址}
    images_url: dict[str, str] = upload_to_minio_return_url(image_context_list, md_path_obj.stem)
    # 6.使用正则进行md_content内容的替换(md_content,{图片名：语义...})--->变成新的md_content
    new_md_content: str = replace_old_md_content(md_content, images_summaries, images_url)
    # 7.对新的md_content进行备份(md_content,md_path_obj) ->new_md_content地址
    new_md_path_obj = backup_new_md_content(new_md_content, md_path_obj)
    # 8.更新state
    #8.1 更新state md_content md_path_obj
    state['md_content'] = new_md_content
    state['md_path_obj'] = new_md_path_obj
    #8.2返回state
    return state

"""
Markdown
 |
 |
extract_image_context_info
 |
 |
找到图片+上下文
 |
 |
Qwen2.5-VL
 |
 |
images_summaries
 |
 |
MinIO上传
 |
 |
images_url
 |
 |
replace_old_md_content
 |
 |
backup_new_ds_content
 |
 |
新的Markdown
"""
