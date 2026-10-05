# -*- coding: utf-8 -*-
"""
电商意图三分类 · BERT 微调脚本

数据：classify_data/intent_train.jsonl（由 generate_data.py 生成；缺失时回退 intent_seed.jsonl）
产物：models/intent_bert  —— routing/intent_classifier.py 启动时自动加载（3 分类即启用）

用法：
    python routing/train/train_intent.py
    python routing/train/train_intent.py --epochs 3 --batch 16 --lr 2e-5
"""
import argparse
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from base.config import config
from base.logger import logger

# ★★ 顺序必须与 routing/intent_classifier.py 的 LABELS 完全一致 ★★
LABELS = ["闲聊", "产品咨询", "售后政策"]
LABEL2ID = {lb: i for i, lb in enumerate(LABELS)}


class IntentDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("query") and o.get("label") in LABEL2ID:
                rows.append(o)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(config.PROJECT_ROOT, "classify_data", "intent_train.jsonl"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--out", default=os.path.join(config.MODELS_DIR, "intent_bert"))
    args = ap.parse_args()

    # 数据：优先用生成数据，缺失则回退种子数据
    data_path = args.data
    if not os.path.exists(data_path):
        seed = os.path.join(config.PROJECT_ROOT, "classify_data", "intent_seed.jsonl")
        logger.warning(f"未找到 {data_path}，回退种子数据 {seed}")
        data_path = seed
    data = load_jsonl(data_path)
    if len(data) < 30:
        logger.error(f"训练数据太少（{len(data)} 条），请先运行 generate_data.py 或补充种子数据")
        return
    texts = [d["query"] for d in data]
    labels = [LABEL2ID[d["label"]] for d in data]
    logger.info(f"加载 {len(data)} 条；分布 {[(lb, labels.count(i)) for i, lb in enumerate(LABELS)]}")

    # 划分 8:2（分层，保证每类都有）
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)

    pretrained = os.path.join(config.MODELS_DIR, "bert-base-chinese")
    tokenizer = BertTokenizer.from_pretrained(pretrained)

    def encode(ts):
        return tokenizer(ts, truncation=True, padding=True, max_length=args.max_len, return_tensors="pt")

    train_ds = IntentDataset(encode(train_texts), train_labels)
    val_ds = IntentDataset(encode(val_texts), val_labels)

    device_is_cuda = torch.cuda.is_available() and getattr(config, "MODEL_DEVICE", "auto") != "cpu"
    model = BertForSequenceClassification.from_pretrained(pretrained, num_labels=len(LABELS))
    # 显式写死 id2label，保证线上 predict 拿到正确顺序
    model.config.id2label = {i: lb for i, lb in enumerate(LABELS)}
    model.config.label2id = LABEL2ID

    def compute_metrics(eval_pred):
        logits, labs = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labs, preds)}

    # 兼容不同 transformers 版本的参数名（eval_strategy / evaluation_strategy）
    common = dict(
        output_dir=os.path.join(config.PROJECT_ROOT, "routing", "train", "bert_results"),
        save_total_limit=1,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=os.path.join(config.PROJECT_ROOT, "routing", "train", "bert_logs"),
        logging_steps=20,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=device_is_cuda,
        report_to="none",
    )
    try:
        training_args = TrainingArguments(eval_strategy="epoch", **common)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="epoch", **common)

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_ds, eval_dataset=val_ds,
                      compute_metrics=compute_metrics)

    logger.info("开始训练 ...")
    trainer.train()

    # 评估：准确率 + 每类 P/R/F1
    pred = trainer.predict(val_ds)
    preds = np.argmax(pred.predictions, axis=-1)
    logger.info("\n" + classification_report(val_labels, preds, target_names=LABELS, digits=3))

    # 保存 → 线上自动加载
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    logger.info(f"模型已保存到 {args.out}；重启 app 即自动启用三分类")


if __name__ == "__main__":
    main()
