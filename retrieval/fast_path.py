"""
多级前置链路（快路径）：正则 → Redis缓存 → MySQL BM25 → RAG 兜底
对齐实战口径："高频问题不配走全链路" —— 省钱省时。
"""
import re
import numpy as np
from rank_bm25 import BM25Okapi
from jieba import lcut

from base.config import config
from base.logger import logger
from retrieval.mysql_client import MySQLClient
from retrieval.redis_cache import RedisClient


# ---------- 第 0 级：规则/正则（问候、可规范化的问题） ----------
RULES = [
    # (正则, 回复) 示例：欢迎语/告别/物流模板等
    (r"^(你好|您好|hi|hello)", "您好！我是**数码商城**智能客服。您可以问我产品参数、价格、保修、退换货等问题。"),
    (r"^(在吗|在不在|有人吗)", "在的，请问有什么可以帮您？"),
    (r"^再见|拜拜|谢谢", "不客气，祝您购物愉快！"),
]


def match_rule(query: str):
    for pat, reply in RULES:
        if re.match(pat, query.strip(), re.IGNORECASE):
            return reply
    return None


# ---------- 第 1 级：Redis 缓存 ----------
# ---------- 第 2 级：MySQL BM25 ----------
def preprocess(text: str):
    return lcut(text.lower())


class FastPath:
    """三明治快路径：规则 → Redis → BM25(MySQL)。未命中返回 need_rag=True。"""

    def __init__(self, mysql: MySQLClient, redis: RedisClient):
        self.mysql = mysql
        self.redis = redis
        self._load_bm25()

    def _load_bm25(self):
        key_o, key_t = "qa_original", "qa_tokenized"
        original = self.redis.get_json(key_o)
        tokenized = self.redis.get_json(key_t)
        if not original or not tokenized:
            original = self.mysql.fetch_all_questions()
            tokenized = [preprocess(q) for q in original]
            self.redis.set_json(key_o, original, ttl=3600)
            self.redis.set_json(key_t, tokenized, ttl=3600)
        self.orig = original
        self.bm25 = BM25Okapi(tokenized) if tokenized else None
        logger.info(f"BM25 初始化完成，共 {len(original)} 个高频问题")

    def search(self, query):
        """返回 (answer, need_rag)。命中直接在规则/缓存/BM25 里返回。"""
        # 规则
        rule_reply = match_rule(query)
        if rule_reply:
            return rule_reply, False
        # Redis
        cached = self.redis.get_answer(query)
        if cached:
            return cached, False
        # BM25
        if self.bm25 and self.orig:
            scores = self.bm25.get_scores(preprocess(query))
            exp = np.exp(scores - scores.max())
            softmax = exp / exp.sum()
            best = int(softmax.argmax())
            if softmax[best] >= config.BM25_THRESHOLD:      # 0.85
                question = self.orig[best]
                answer = self.mysql.fetch_answer(question)
                if answer:
                    self.redis.set_answer(query, answer)    # 回写缓存，下次 0 token
                    return answer, False
        return None, True