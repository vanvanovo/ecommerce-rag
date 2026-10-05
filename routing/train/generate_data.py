# -*- coding: utf-8 -*-
"""
调用阿里百炼(Qwen) 批量生成三分类训练数据 → classify_data/intent_train.jsonl

用法：
    python routing/train/generate_data.py --per-class 1500
    # 生成约 4500 条（3 类 × 1500），写入 classify_data/intent_train.jsonl
"""
import argparse
import json
import os
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openai import OpenAI
from base.config import config
from base.logger import logger
from routing.train.prompts import SYSTEM_PROMPT, DATA_GEN_PROMPT

LABELS = ["闲聊", "产品咨询", "售后政策"]
OUT_PATH = os.path.join(config.PROJECT_ROOT, "classify_data", "intent_train.jsonl")


def parse_rows(text):
    rows = []
    for line in text.splitlines():
        line = line.strip().strip("`").strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        q, lb = obj.get("query"), obj.get("label")
        if isinstance(q, str) and q.strip() and lb in LABELS:
            rows.append({"query": q.strip(), "label": lb})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1500, help="每类目标条数")
    ap.add_argument("--batch", type=int, default=150, help="每次请求生成的条数（每类）")
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    client = OpenAI(api_key=config.DASHSCOPE_API_KEY, base_url=config.DASHSCOPE_BASE_URL)
    seen = set()
    rows = []
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # 已存在则续跑，去重
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                    if o.get("query") not in seen:
                        seen.add(o["query"]); rows.append(o)
                except Exception:
                    pass
        logger.info(f"续跑：已有 {len(rows)} 条")

    def count(label):
        return sum(1 for r in rows if r["label"] == label)

    round_no = 0
    while any(count(lb) < args.per_class for lb in LABELS):
        round_no += 1
        need = max(args.per_class - count(lb) for lb in LABELS)
        per = min(args.batch, max(need, 20))
        logger.info(f"第 {round_no} 轮：每类目标 {per} 条，当前 {[(lb, count(lb)) for lb in LABELS]}")
        try:
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": DATA_GEN_PROMPT.format(per_class=per)},
                ],
                temperature=args.temperature,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM 生成失败: {e}；10 秒后重试")
            time.sleep(10)
            continue

        added = 0
        for r in parse_rows(text):
            if r["query"] not in seen:
                seen.add(r["query"]); rows.append(r); added += 1
        logger.info(f"本轮新增 {added} 条，累计 {len(rows)} 条")

        # 增量落盘，防中断丢失
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        if added == 0:
            logger.warning("本轮无新增，可能已收敛或模型未按格式输出；继续重试")

    logger.info(f"完成：共 {len(rows)} 条 → {OUT_PATH}")
    for lb in LABELS:
        logger.info(f"  {lb}: {count(lb)} 条")


if __name__ == "__main__":
    main()
