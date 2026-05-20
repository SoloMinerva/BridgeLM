# Configs

每个 JSON 对应一次训练运行，通过 `--config` 参数传入：

```bash
python scripts/train_pretrain.py --config configs/pretrain_full.json
```

## 快速开始（smoke，CPU 可跑）

| Config | 脚本 | 用途 |
|---|---|---|
| pretrain_smoke.json | train_pretrain.py | 验证预训练流程 |修正
| sft_smoke.json | train_sft.py | 验证 SFT 流程 |
| tokenizer_smoke.json | train_tokenizer.py | 验证分词器训练 |
| qwen_lora_fake_smoke.json | train_qwen_lora.py | 验证 Qwen LoRA 流程 |

## 完整训练（需要 GPU）

| Config | 脚本 | 说明 |
|---|---|---|
| pretrain_full.json | train_pretrain.py | 自研预训练，20000 iter |
| pretrain_full_corpus.json | train_pretrain.py | 自研预训练，50000 iter |
| sft_baseline.json | train_sft.py | 自研 SFT baseline，1000 step batch=2 |
| sft_smoke_test.json | train_sft.py | 自研 SFT smoke，50 step，验证 improved 流程 |
| sft_1k_b8.json | train_sft.py | 自研全参 SFT，1000 step batch=8（对比实验） |
| sft_lora_1k_b8.json | train_sft.py | 自研 LoRA SFT，1000 step batch=8（对比实验） |
| tokenizer_full_clean.json | train_tokenizer.py | 完整分词器训练 |
| tokenize_full_corpus.json | tokenize_corpus_fast.py | 全量 token id 编码 |
| qwen_lora_structured_smoke.json | train_qwen_lora.py | Qwen LoRA smoke，50 step |
| qwen_lora_structured.json | train_qwen_lora.py | Qwen LoRA 完整训练，2000 step |
