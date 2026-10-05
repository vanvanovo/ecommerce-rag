"""
RAG 生成系统（慢路径兜底）——电商版

流程：
  意图(闲聊/产品咨询/售后政策) → 指代消解改写 → 规则+意图驱动策略路由
  → BGE-M3 混合检索+WeightedRanker → BGE-Reranker 精排 → 相关度守门 → Qwen 流式生成

对齐实战口径：
- 政策类内容必须 100% 来自检索上下文（忠实度红线）；
- 检索相关度太低 → 明确告知用户并给人工联系方式，不硬答。
"""
import json
import re
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from base.config import config
from base.logger import logger
from core.vector_store import VectorStore
from routing.intent_classifier import IntentClassifier


# ---------- 导购型 Prompt（产品咨询） ----------
PRODUCT_PROMPT = PromptTemplate.from_template("""
你是数码商城的导购客服。规则：
1. 基于【上下文】回答产品的参数、价格、型号、选购等问题，可做**客观对比与推荐**。
2. **禁止编造参数/价格**；引用时注明"根据产品手册/参数表"。
3. 上下文不足时，回复："这个问题我暂时没有查到确切信息，请咨询人工客服：{phone}。"
4. 结合【对话历史】保持多轮一致；历史无关则忽略。

【上下文】
{context}

【对话历史】
{history}

【问题】
{question}

【回答】
""")

# ---------- 严谨政策型 Prompt（售后政策） ----------
POLICY_PROMPT = PromptTemplate.from_template("""
你是数码商城的售后政策专员。规则（红线，必须遵守）：
1. **退换货、保修、价保、发票等政策内容必须 100% 来自【上下文】原文**，一个字都不能自行发挥或推测。
2. 若【上下文】没有覆盖，直接回复："该问题需要人工核实，请联系人工客服：{phone}。" 不要猜测天数或条件。
3. 回答中注明"根据售后政策"。

【上下文】
{context}

【对话历史】
{history}

【问题】
{question}

【回答】
""")

# ---------- 指代消解：把依赖上下文的问题改写成完整独立问题 ----------
REWRITE_PROMPT = PromptTemplate.from_template("""
根据【对话历史】，把【当前问题】改写成不依赖上下文也能看懂的完整问题。
只输出改写后的问题，不要解释、不要引号。

【对话历史】
{history}

【当前问题】
{query}

改写后：
""")

HYDE_PROMPT = PromptTemplate.from_template(
    "假设你是数码商城客服，针对下面这个问题写一段可能出现在产品手册/售后政策里的简答内容（只写内容，不要客套话）：\n{query}\n假设答案：")

SUBQUERY_PROMPT = PromptTemplate.from_template(
    "下面这个电商问题包含多个实体或方面，请拆成每个实体一个子问题，每行一个，不要编号：\n{query}")


class RAGSystem:
    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store
        self.intent = IntentClassifier()
        self.llm = OpenAI(api_key=config.DASHSCOPE_API_KEY, base_url=config.DASHSCOPE_BASE_URL)

    # ---------- 1) 指代消解 / 查询改写 ----------
    def rewrite_query(self, query, history):
        if not history:
            return query
        pronouns = ("它", "他", "她", "这个", "那个", "该", "此", "这", "那")
        need = len(query) <= 8 or any(p in query for p in pronouns)
        if not need:
            return query
        try:
            rewritten = self._llm_text(REWRITE_PROMPT.format(
                history=json.dumps(history, ensure_ascii=False), query=query)).strip()
            if rewritten:
                logger.info(f"指代消解: '{query}' → '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.error(f"指代消解失败，使用原问题: {e}")
        return query

    # ---------- 2) 规则驱动的查询类型判断 ----------
    @staticmethod
    def _is_multi_entity(q):
        return any(k in q for k in ("和", "与", "、", "vs", "VS", "哪个", "还是", "对比", "比较"))

    @staticmethod
    def _is_open(q):
        return any(k in q for k in ("推荐", "适合", "怎么选", "选哪", "有什么好", "哪个好"))

    @staticmethod
    def _is_exact(q):
        # 含型号 token（字母+数字）或明确要参数/价格
        return bool(re.search(r"[A-Za-z]+\s?\d+", q)) or any(
            k in q for k in ("多少钱", "价格", "售价", "报价", "参数", "配置", "尺寸", "重量", "电池", "内存"))

    # ---------- 3) 策略路由 + 检索 ----------
    def retrieve(self, query, source_filter=None):
        # 规则优先：多实体 → 子查询；开放推荐 → HyDE；精确/政策/其他 → 直检
        if self._is_multi_entity(query):
            strategy = "子查询检索"
        elif self._is_open(query):
            strategy = "假设问题检索"
        else:
            strategy = "直接检索"
        logger.info(f"策略路由: '{query}' → {strategy}")

        if strategy == "假设问题检索":
            hypo = self._llm_text(HYDE_PROMPT.format(query=query))
            docs = self.vs.hybrid_search_with_rerank(hypo or query, source_filter=source_filter)
        elif strategy == "子查询检索":
            subs = [s.strip() for s in self._llm_text(SUBQUERY_PROMPT.format(query=query)).split("\n") if s.strip()]
            all_docs = []
            for s in subs:
                all_docs += self.vs.hybrid_search_with_rerank(s, source_filter=source_filter)
            docs = list({d.page_content: d for d in all_docs}.values())[:config.CANDIDATE_M]
        else:
            docs = self.vs.hybrid_search_with_rerank(query, source_filter=source_filter)
        return docs

    # ---------- 4) 流式生成 ----------
    def generate_answer(self, query, source_filter=None, history=None):
        label, prob = self.intent.predict(query)
        logger.info(f"意图: '{query}' → {label} ({prob:.2f})")

        if label == "闲聊" and prob >= 0.9:
            yield "您好，请问需要了解产品参数、价格、保修或售后政策吗？"
            return

        # 指代消解：多轮里"那它保修多久"→ 补全为完整问题
        real_query = self.rewrite_query(query, history)

        # 意图决定 source_filter 与 Prompt
        if label == "售后政策":
            source_filter = source_filter or "售后政策"
            prompt_template = POLICY_PROMPT
        else:
            prompt_template = PRODUCT_PROMPT

        docs = self.retrieve(real_query, source_filter) or []
        context = "\n\n".join(d.page_content for d in docs) if docs else ""
        if not context.strip():
            # 相关度守门：没召回到 → 不硬答
            yield f"抱歉，我没有在知识库中找到相关信息，请咨询人工客服：{config.CUSTOMER_SERVICE_PHONE}。"
            return

        prompt = prompt_template.format(
            context=context, history=json.dumps(history or [], ensure_ascii=False),
            question=query, phone=config.CUSTOMER_SERVICE_PHONE)
        for chunk in self._stream_llm(prompt):
            yield chunk

    # ---------- LLM 工具 ----------
    def _llm_text(self, prompt):
        try:
            return self.llm.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_think": False},
            ).choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return ""

    def _stream_llm(self, prompt):
        try:
            completion = self.llm.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                extra_body={"enable_think": False},
            )
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            yield f"抱歉，生成过程出错，请联系人工客服：{config.CUSTOMER_SERVICE_PHONE}。"