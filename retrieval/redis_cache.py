"""Redis 缓存：BM25 索引（分词后问题） + 热点「问题→答案」。"""
import json
import redis
from base.config import config
from base.logger import logger


class RedisClient:
    def __init__(self):
        self.client = redis.StrictRedis(
            host=config.REDIS_HOST, port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD, db=config.REDIS_DB,
            decode_responses=True)
        self.client.ping()
        logger.info("Redis 连接成功")

    def get_json(self, key):
        data = self.client.get(key)
        return json.loads(data) if data else None

    def set_json(self, key, value, ttl=None):
        kwargs = {"ex": ttl} if ttl else {}
        self.client.set(key, json.dumps(value, ensure_ascii=False), **kwargs)

    # 热点问答：同一个问题直接回答案（命中=0 token）
    def get_answer(self, query):
        return self.get_json(f"answer:{query}")

    def set_answer(self, query, answer, ttl=3600):
        self.set_json(f"answer:{query}", answer, ttl=ttl)