# Data

本项目使用 **两个外部数据源**，分别服务于两条训练链路。

---

## 数据源总览

| 数据源 | 用途 | 服务链路 | 原始规模 | 处理后规模 |
|--------|------|----------|----------|------------|
| [MiniMind](#1-minimind) | 预训练 + SFT 对话微调 | 自研 BridgeLM（自研线） | ~127 万条 | Pretrain: 125万 / SFT: 90万 |
| [InstructIE](#2-instructie) | 结构化信息抽取 SFT | Qwen 迁移线（工业线） | 171,471 条 | 28.5K train + 1.5K valid |

---

## 1. MiniMind

**用途**：自研 BridgeLM 的预训练语料与 SFT 对话数据。

### 1.1 预训练语料

- **格式**：每行 `{"text": "..."}`，中文对话/指令/知识文本
- **原始规模**：**1,270,238 条**（约 **1.24 GB**）
- **来源**：HuggingFace — [`jingyaogong/minimind_dataset`](https://huggingface.co/datasets/jingyaogong/minimind_dataset)
- **下载方式**：

```python
from huggingface_hub import hf_hub_download
import os

# 国内用户加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

hf_hub_download(
    repo_id="jingyaogong/minimind_dataset",
    repo_type="dataset",
    filename="pretrain_t2t_mini.jsonl",
    local_dir="data",
)
```

- **处理流程**（`scripts/prepare_pretrain_jsonl.py`）：

```
pretrain_t2t_mini.jsonl (1,270,238 条)
  → 控制字符清理 + HTML 标签清理 + 空白压缩
  → 长度过滤 + SHA256 精确去重
  → SHA1 哈希确定性划分 train/valid (99:1)
  ↓
pretrain_clean/
  ├── train.txt          (1,251,547 条)
  ├── valid.txt          (12,504 条)
  └── tokenizer_corpus.txt
```

清洗统计：总过滤率 0.49%（HTML 清理 / 去重 / 长度过滤）

### 1.2 SFT 对话数据

- **文件**：`data/sft_t2t_mini.jsonl`
- **规模**：905,718 条 / 1.6 GB
- **用途**：BridgeLM SFT 全参微调训练数据
- **下载方式**：

```python
from huggingface_hub import hf_hub_download
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

hf_hub_download(
    repo_id="jingyaogong/minimind_dataset",
    repo_type="dataset",
    filename="sft_t2t_mini.jsonl",
    local_dir="data",
)
```

- **验证集**：`data/sft_valid.jsonl`（从 sft_t2t_mini.jsonl 最后 1% 切分，9,057 条）

---

## 2. InstructIE

**用途**：Qwen2.5-1.5B-Instruct 结构化信息抽取 LoRA 微调。

- **来源**：HuggingFace — [`zjunlp/InstructIE`](https://huggingface.co/datasets/zjunlp/InstructIE)
- **原始规模**：train **171,471** 条 / valid 1,004 条 / test 1,002 条
- **覆盖主题**：12 个（人物、组织、地点、事件、作品、医学、自然科学等）
- **下载方式**：

```python
from huggingface_hub import hf_hub_download
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

for filename in ["train_zh.json", "valid_zh.json", "test_zh.json", "schema_zh.json"]:
    hf_hub_download(
        repo_id="zjunlp/InstructIE",
        repo_type="dataset",
        filename=filename,
        local_dir="data/instructie",
    )
```

### 处理 Pipeline（6 步）

```
InstructIE 原始数据 (171,471 条)
  │
  ├─ Step 1: 01_normalize.py      字段标准化
  ├─ Step 2: 02_filter.py         硬过滤 + P99 软过滤 → 163,629 条
  ├─ Step 3: 03_quality_tier.py   质量分层 (high 95.5% / medium / low)
  ├─ Step 4: 04_derive_tasks.py   四类任务派生 → 623,650 条
  │          ie_extraction(50%) / text_to_json(25%)
  │          format_following(15%) / schema_repair(10%)
  ├─ Step 5: 05_stratified_sample.py  分层采样 → 30,000 条
  └─ Step 6: 06_to_chat_jsonl.py  格式转写 + valid 切分 (5%)
  ↓
sft_candidate/
  ├── train.jsonl   (28,500 条)
  ├── valid.jsonl   (1,500 条)
  └── metadata.json
```

所有阈值集中配置在 `scripts/conf.py`。

---

## 目录结构

```
data/
├── pretrain_clean/            # 清洗后的预训练文本
│   ├── train.txt              # 1,251,547 条
│   ├── valid.txt              # 12,504 条
│   ├── tokenizer_sample.txt   # 分词器训练采样语料 (15MB)
│   └── tokenized_full/        # BPE 编码后的 token ids
│       ├── train_ids.npy      # 3.23 亿 tokens (617MB)
│       └── valid_ids.npy
├── instructie/                # InstructIE 原始数据
├── processed/                 # 6 步 pipeline 中间产物
├── sft_candidate/             # 最终 SFT 数据 (28.5K train + 1.5K valid)
├── sft_t2t_mini.jsonl         # BridgeLM SFT 训练数据 (90万条)
├── sft_valid.jsonl            # BridgeLM SFT 验证集 (9,057 条)
├── smoke/                     # 预训练 smoke 数据
├── sft_smoke/                 # 自研线 SFT smoke 数据
└── sft_smoke_fake/            # Qwen LoRA smoke 数据 (假数据)
```

---

## 引用

- **MiniMind**：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)
- **InstructIE**：Wang, Y. et al. *InstructIE: A Bilingual Instruction-based Information Extraction Dataset* — [zjunlp/InstructIE](https://huggingface.co/datasets/zjunlp/InstructIE)
