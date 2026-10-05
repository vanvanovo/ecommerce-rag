"""
BERT 意图分流（电商三分类）：闲聊 / 产品咨询 / 售后政策

- 有微调好的 `models/intent_bert`（3 分类）时用模型；
- 还没训好时退化为关键词规则，保证骨架端到端可跑（训练后自动启用模型）。
训练参考：edu_rag_pro/rag_qa/core/query_classifier.py（LLM 造数据 → Trainer 微调 bert-base-chinese）。
设备走 config.MODEL_DEVICE（auto/cpu/cuda/mps）。
"""
import os
import sys

# 允许直接用 `python routing/intent_classifier.py` 运行
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.config import config
from base.logger import logger

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    from transformers import BertTokenizer, BertForSequenceClassification
    _HAS_TF = True
except Exception:
    _HAS_TF = False


class IntentClassifier:
    LABELS = ["闲聊", "产品咨询", "售后政策"]

    # 售后/政策关键词（命中即判"售后政策"，优先于产品）
    POLICY_KW = ["退货", "退换", "换货", "保修", "质保", "维修", "发票", "价保", "补差",
                 "运费", "退款", "售后", "激活", "七天", "7天", "无理由", "三包", "人工", "客服电话"]
    # 闲聊关键词
    CHITCHAT_KW = ["你好", "您好", "hi", "hello", "在吗", "在不在", "有人吗",
                   "再见", "拜拜", "谢谢", "哈喽", "辛苦了"]

    def __init__(self, model_path=None):
        self.pretrained = os.path.join(config.MODELS_DIR, 'bert-base-chinese')
        self.model_path = model_path or os.path.join(config.MODELS_DIR, 'intent_bert')
        self.device = self._resolve_device()
        self.model = None
        self.tokenizer = None
        self._load_model()

    @staticmethod
    def _resolve_device():
        if not _HAS_TORCH:
            return 'cpu'
        want = getattr(config, 'MODEL_DEVICE', 'auto')
        if want in ('cpu', 'cuda', 'mps'):
            return want
        if torch.cuda.is_available():
            return 'cuda'
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def _load_model(self):
        if not (_HAS_TF and _HAS_TORCH and os.path.exists(self.model_path)):
            logger.warning("未找到电商三分类意图模型，使用关键词规则兜底（训练后自动启用）")
            return
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            model = BertForSequenceClassification.from_pretrained(self.model_path)
            if model.config.num_labels != len(self.LABELS):
                logger.warning(
                    f"意图模型标签数={model.config.num_labels}，与电商三分类({len(self.LABELS)})不匹配，"
                    f"改用规则兜底；请重新微调 intent_bert")
                return
            self.model = model.to(self.device).eval()
            logger.info(f"意图分类器(模型)就绪，设备={self.device}，标签={self.LABELS}")
        except Exception as e:
            logger.error(f"加载意图模型失败，改用规则兜底: {e}")
            self.model = None

    # ---------- 规则兜底 ----------
    def _rule_predict(self, query):
        q = query.lower().strip()
        if any(q.startswith(kw) for kw in self.CHITCHAT_KW):
            return "闲聊", 0.99
        if any(kw in q for kw in self.POLICY_KW):
            return "售后政策", 0.90
        return "产品咨询", 0.70     # 默认归产品（电商绝大多数是产品咨询）

    # ---------- 对外 ----------
    def predict(self, query):
        if self.model is None:
            return self._rule_predict(query)
        enc = self.tokenizer(query, truncation=True, padding=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
            prob = torch.softmax(logits, dim=-1)[0]
            idx = int(torch.argmax(prob))
        return self.LABELS[idx], float(prob[idx].item())

    def is_chitchat(self, query, threshold=0.9):
        label, prob = self.predict(query)
        return label == "闲聊" and prob >= threshold


if __name__ == '__main__':
    ic = IntentClassifier()
    for q in ["你好呀", "iPhone16 Pro 128G 多少钱", "屏幕摔了保修吗", "7天无理由怎么退", "拍照好的手机推荐"]:
        print(q, "->", ic.predict(q))