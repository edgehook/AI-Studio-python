import logging

LOGGING_NAME = "AI-Studio"
logging.basicConfig(level=logging.DEBUG,  # 设置日志级别
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
                    datefmt='%Y-%m-%d %H:%M:%S')  # 设置日期格式
 
# 创建一个日志记录器对象
logger = logging.getLogger(LOGGING_NAME)