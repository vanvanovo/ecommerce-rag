"""日志：控制台 + 文件（RotatingFileHandler 可防日志无限膨胀）"""
import logging
import os
from logging.handlers import RotatingFileHandler
from base.config import config


def setup_logger(name='EcomRAG', file=config.LOG_FILE):
    dirname = os.path.dirname(file)
    os.makedirs(dirname, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(pathname)s - %(funcName)s - %(lineno)d - %(message)s')
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(fmt)
        fh = RotatingFileHandler(file, mode='a', maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger


logger = setup_logger()