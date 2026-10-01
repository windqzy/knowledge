
from app.shared.runtime.logger import logger
from app.shared.runtime.load_prompt import load_prompt

image_str:str = load_prompt(name="image_summary",root_folder="images",image_content=("1111","2222"))
logger.debug("加载的提示词:{}",image_str)