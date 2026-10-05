# -*- coding: utf-8 -*-
"""
意图分类 · 评测集回归脚本（分层统计 + 基线门禁）

用途：每次改动意图分类（换模型/改规则/加类别）后，一键跑评测集，出指标并与基线对比，
      分数不降才算通过（用退出码当门禁，可接 CI）。

用法：
    python routing/train/evaluate_intent.py                    # 评测 + 与基线对比
    python routing/train/evaluate_intent.py --save-baseline    # 把本次存为基线
    python routing/train/evaluate_intent.py --min-accuracy 0.9 --max-f1-drop 0.02
    python routing/train/evaluate_intent.py --model-path models/intent_bert

退出码：0=通过；1=未达门槛或低于基线
"""
import argparse
import csv
import datetime
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from base.config import config
from base.logger import logger
from routing.intent_classifier import IntentClassifier

LABELS = IntentClassifier.LABELS
DEFAULT_DATA = os.path.join(config.PROJECT_ROOT, "classify_data", "intent_eval.jsonl")
DEFAULT_BASELINE = os.path.join(config.PROJECT_ROOT, "routing", "train", "baseline_intent.json")
DEFAULT_OUT = os.path.join(config.PROJECT_ROOT, "results")


# ---------- 数据 ----------
def load_eval(path):
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
            if o.get("query") and o.get("label") in LABELS:
                o.setdefault("type", "未标注")
                rows.append(o)
    return rows


# ---------- 指标（手写，免 sklearn 依赖） ----------
def compute_metrics(y_true, y_pred):
    cm = {lt: {lp: 0 for lp in LABELS} for lt in LABELS}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    per = {}
    for lb in LABELS:
        tp = cm[lb][lb]
        fp = sum(cm[o][lb] for o in LABELS if o != lb)
        fn = sum(cm[lb][o] for o in LABELS if o != lb)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[lb] = {"precision": prec, "recall": rec, "f1": f1, "support": sum(cm[lb].values())}
    acc = sum(cm[l][l] for l in LABELS) / len(y_true) if y_true else 0.0
    macro_f1 = sum(per[l]["f1"] for l in LABELS) / len(LABELS)
    return acc, macro_f1, per, cm


def stratified_accuracy(records):
    """按 type 分组准确率，定位弱项。"""
    by = {}
    for r in records:
        d = by.setdefault(r["type"], [0, 0])
        d[1] += 1
        if r["true"] == r["pred"]:
            d[0] += 1
    return {t: (c / n if n else 0.0, n) for t, (c, n) in by.items()}


# ---------- 报告 ----------
def print_report(acc, macro_f1, per, cm, strat, conf_low_rate):
    line = "=" * 64
    print("\n" + line)
    print("意图分类评测报告")
    print(line)
    print(f"整体准确率 accuracy : {acc:.4f}")
    print(f"宏平均           F1 : {macro_f1:.4f}")
    print(f"低置信率 (<0.6)     : {conf_low_rate:.4f}")
    print("-" * 64)
    print(f"{'类别':<10}{'precision':>12}{'recall':>12}{'f1':>12}{'support':>10}")
    for lb in LABELS:
        m = per[lb]
        print(f"{lb:<10}{m['precision']:>12.3f}{m['recall']:>12.3f}{m['f1']:>12.3f}{m['support']:>10}")
    print("-" * 64)
    print("混淆矩阵（行=真实，列=预测）")
    print(f"{'':<10}" + "".join(f"{lb:>10}" for lb in LABELS))
    for lt in LABELS:
        print(f"{lt:<10}" + "".join(f"{cm[lt][lp]:>10}" for lp in LABELS))
    print("-" * 64)
    print("分层准确率（按 query 类型）")
    for t, (a, n) in sorted(strat.items(), key=lambda x: x[1][0]):
        print(f"  {t:<8} acc={a:.3f}  (n={n})")
    print(line)


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model-path", default=None, help="不传则用默认 models/intent_bert；不存在自动规则兜底")
    ap.add_argument("--baseline", default=DEFAULT_BASELINE)
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--min-accuracy", type=float, default=0.0)
    ap.add_argument("--max-f1-drop", type=float, default=0.02)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = load_eval(args.data)
    if not rows:
        logger.error(f"评测集为空或格式错误: {args.data}")
        sys.exit(2)

    clf = IntentClassifier(model_path=args.model_path)
    used_model = clf.model is not None

    records = []
    for r in rows:
        pred, conf = clf.predict(r["query"])
        records.append({"query": r["query"], "true": r["label"], "pred": pred,
                        "conf": conf, "type": r["type"],
                        "ok": pred == r["label"]})
    y_true = [r["true"] for r in records]
    y_pred = [r["pred"] for r in records]

    acc, macro_f1, per, cm = compute_metrics(y_true, y_pred)
    strat = stratified_accuracy(records)
    conf_low_rate = sum(1 for r in records if r["conf"] < 0.6) / len(records)

    print(f"\n数据: {args.data}（{len(records)} 条） | 预测器: {'模型' if used_model else '规则兜底'}")
    print_report(acc, macro_f1, per, cm, strat, conf_low_rate)

    # 结果落盘
    os.makedirs(args.out, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": ts, "data": os.path.basename(args.data), "n": len(records),
        "used_model": used_model, "accuracy": acc, "macro_f1": macro_f1,
        "low_conf_rate": conf_low_rate, "per_class": per,
        "stratified": {t: {"acc": a, "n": n} for t, (a, n) in strat.items()},
    }
    with open(os.path.join(args.out, f"intent_eval_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # badcase 台账
    bad = [r for r in records if not r["ok"]]
    if bad:
        bc_path = os.path.join(args.out, f"intent_badcase_{ts}.csv")
        with open(bc_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["query", "true", "pred", "conf", "type"])
            w.writeheader()
            for r in bad:
                w.writerow({k: r[k] for k in ["query", "true", "pred", "conf", "type"]})
        logger.info(f"badcase 已导出: {bc_path}（{len(bad)} 条）")

    # 基线门禁
    if args.save_baseline:
        with open(args.baseline, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存基线: {args.baseline}")
        print("\n[基线已保存] 退出码 0")
        sys.exit(0)

    failed = []
    if acc < args.min_accuracy:
        failed.append(f"准确率 {acc:.4f} < 门槛 {args.min_accuracy:.4f}")
    if os.path.exists(args.baseline):
        base = json.load(open(args.baseline, encoding="utf-8"))
        drop = base.get("macro_f1", 0) - macro_f1
        print(f"\n基线对比: macro_f1 {base.get('macro_f1', 0):.4f} → {macro_f1:.4f} (变化 {macro_f1 - base.get('macro_f1', 0):+.4f})")
        if drop > args.max_f1_drop:
            failed.append(f"macro_f1 下降 {drop:.4f} > 允许 {args.max_f1_drop:.4f}")
        for lb in LABELS:
            b_f1 = base.get("per_class", {}).get(lb, {}).get("f1")
            if b_f1 is not None and (b_f1 - per[lb]["f1"]) > args.max_f1_drop:
                failed.append(f"类别[{lb}] F1 下降 {b_f1 - per[lb]['f1']:.4f}")
    else:
        print(f"\n未找到基线（{args.baseline}），本次可先跑 --save-baseline 建立基线")

    print("\n" + "-" * 64)
    if failed:
        print("[不通过] 回归门禁未达标：")
        for x in failed:
            print("  - " + x)
        sys.exit(1)
    print("[通过] 回归门禁达标")
    sys.exit(0)


if __name__ == "__main__":
    main()
