# 电商意图三分类 · 训练闭环

标签定义（顺序必须与 `routing/intent_classifier.py` 的 `LABELS` 一致）：
`["闲聊", "产品咨询", "售后政策"]`

## 流程

```
① 造数据  python routing/train/generate_data.py --per-class 1500
          → classify_data/intent_train.jsonl（约 4500 条，三类均衡）

② 微调    python routing/train/train_intent.py
          → models/intent_bert（标签顺序已锁死）

③ 上线    重启服务（python -m app.main）
          → IntentClassifier 检测到 3 分类模型，自动启用（不再用规则兜底）
```

> 没有 API Key 也能先跑：`generate_data.py` 生成的 `intent_train.jsonl` 不存在时，
> `train_intent.py` 会自动回退 `classify_data/intent_seed.jsonl`（45 条种子），用于冒烟验证。

## 评估与迭代

- 训练结束会打印每类 **精确率/召回率/F1**（`classification_report`）。
- 线上 badcase 处理：把判错的问题补进对应类别，重跑 ②（**评测集回归：分数不降才合入**）。
- 数据规模参考：每个类别几百到几千条；冷启动可用 LLM 造数据 + 甲方历史客服会话标注。

## 常见坑

1. **标签顺序**：`LABEL2ID` 与 `intent_classifier.LABELS` 不一致 → 分类全错。脚本里已注释提醒。
2. **版本参数**：`transformers` 新旧版本 `eval_strategy` / `evaluation_strategy` 名字不同，脚本已做兼容。
3. **设备**：`fp16` 只在 CUDA 上开；CPU 训练会很慢，建议 GPU 或用小模型/减 epoch。
4. **配比**：三类别失衡，否则模型会偏向多数类（真机实测：售后问题占比高时最容易误判产品咨询）。

## 设计口径对齐

- 一期 RAG：BERT 挡闲聊（二分类起步），本文档扩成电商三分类，是自然的工程升级；
- 二期客服平台：四类业务意图 + 闲聊，可在此脚本基础上直接扩标签；
- 训练数据来源：**甲方历史客服会话标注（冷启动几千条 + 线上 badcase 补标）**。

---

## 评测集回归（evaluate_intent.py）

"评测就是 AI 项目的回归测试"：任何改动（换模型/改规则/加类别）后，跑评测集，**分数不降才合入**。

```powershell
# 第一次：建立基线（当前是规则兜底）
python routing/train/evaluate_intent.py --save-baseline

# 日常：评测 + 与基线对比（门禁：macro_f1 下降超阈值 → 退出码 1）
python routing/train/evaluate_intent.py

# 自定义门槛
python routing/train/evaluate_intent.py --min-accuracy 0.9 --max-f1-drop 0.02
```

**输入**：`classify_data/intent_eval.jsonl`（120 条，含 `type` 字段；独立于训练集防泄露）

**输出**：
- 控制台报告：准确率 / macro-F1 / 每类 P-R-F1 / 混淆矩阵 / **按 type 分层准确率**
- `results/intent_eval_*.json`：结果快照
- `results/intent_badcase_*.csv`：误判台账（query/真实/预测/置信度/type），直接补标

**分层 type**：正常 / 口语 / 错别字 / 多轮指代 / 边界易混 —— 定位弱项用。

**退出码**：0=通过；1=未达门槛或低于基线（可直接接 CI / 合并前门禁）。

**基线现状（规则兜底）**：accuracy 0.683、macro-F1 0.651；弱项在错别字(0.55)/边界易混(0.57)/闲聊召回(0.33)。
训练出 `intent_bert` 后重跑，应有明显提升 —— 这就是"分数不降才合入"的基准。

### 扩评测集与版本管理
- 新发现的线上 badcase：贴进 `intent_eval.jsonl` 并标注正确 label/type（评测集只增不减）。
- 基线文件 `routing/train/baseline_intent.json` 也进 Git，团队共用同一把尺子。
