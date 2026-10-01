from app.shared.runtime.logger import logger #配置好的输出对象



logger.debug("输出的内容debug")

param ="xxx"
age   = 18
logger.info(f"输出的内容info:{param}")
logger.info("输出的内容info:{},我的年龄:{}",param,age)

logger.warning("输出的内容warning")
logger.error("输出的内容error")

try:
    i = 1/0
except Exception as e:
    logger.exception(f"报错了{str(e)}")