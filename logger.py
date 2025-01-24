import logging
from logging.handlers import RotatingFileHandler

# 创建一个 Logger 对象
logger = logging.getLogger(__name__)
# 设置日志级别
logger.setLevel(logging.DEBUG)

# 创建一个 RotatingFileHandler，设置最大文件大小为 1MB，保留 3 个备份文件
handler = RotatingFileHandler('ai-studio.log', maxBytes=1024*1024*10, backupCount=3)
# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)