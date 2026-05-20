"""LLM-as-Judge: use DeepSeek to score model outputs on 4 dimensions.

Usage:
    # Step 1: generate answers from models
    python scripts/run_eval_prompts.py \
        --models sft=outputs/sft_full/ckpt_final.pt \
        --eval-file eval/prompts_v1.json \
        --out-dir results/judge_eval

    # Step 2: DeepSeek scoring
    python scripts/judge_eval.py \
        --results results/judge_eval/eval_results.json \
        --api-key sk-your-deepseek-key \
        --out results/judge_eval/scores.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI

JUDGE_PROMPT = """你是一个严格但公平的 LLM 评估专家。请根据以下 4 个维度为回答评分（每项 1-5 分，可用小数如 3.5）：

1. 相关性 (1-5)：回答是否紧扣问题，有没有跑题
2. 流畅性 (1-5)：语言是否自然通顺、无病句、无重复
3. 准确性 (1-5)：事实是否正确（常识题重点看这个）
4. 完整性 (1-5)：回答是否完整，有没有半截断掉、关键信息遗漏

评分标准：
  5 = 优秀，几乎没有可指摘的
  4 = 良好，有个别小瑕疵
  3 = 及格，大致能接受
  2 = 较差，存在明显问题
  1 = 很差，答非所问或完全不通

请只输出一个 JSON，不要有任何其他文字：
{"相关性": 分数, "流畅性": 分数, "准确性": 分数, "完整性": 分数, "备注": "一句话说明扣分原因或亮点"}"""

JUDGE_PROMPT_SMALL_MODEL = """你是一个 LLM 评估专家。被评估的模型是一个**仅有 31.7M 参数的极小语言模型**（对比：GPT-3 有 175B 参数，约为其 1/5000）。请根据这个模型的实际能力范围，用以下 4 个维度评分（每项 1-5 分，可用小数如 3.5）：

1. 相关性 (1-5)：回答是否紧扣问题，有没有跑题
2. 流畅性 (1-5)：考虑到极小模型容易出现轻微重复，适度宽松；但严重的无限循环仍应扣分
3. 准确性 (1-5)：事实是否正确（常识题重点看这个）
4. 完整性 (1-5)：回答是否完整，有没有半截断掉、关键信息遗漏

评分参考（针对 31.7M 小模型）：
  5 = 对小模型而言表现出色
  4 = 基本答对且较流畅，有小问题
  3 = 答案方向正确但质量一般
  2 = 有明显错误或较多重复
  1 = 完全答非所问或输出乱码

请只输出一个 JSON，不要有任何其他文字：
{"相关性": 分数, "流畅性": 分数, "准确性": 分数, "完整性": 分数, "备注": "一句话说明扣分原因或亮点"}"""


def parse_args():
    p = argparse.ArgumentParser(description="DeepSeek-as-Judge scoring")
    p.add_argument("--results", type=Path, required=True,
                   help="Path to eval_results.json from run_eval_prompts.py")
    p.add_argument("--api-key", type=str, required=True,
                   help="DeepSeek API key")
    p.add_argument("--base-url", type=str, default="https://api.deepseek.com",
                   help="DeepSeek API base URL")
    p.add_argument("--model", type=str, default="deepseek-chat",
                   help="DeepSeek model name")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--limit", type=int, default=None,
                   help="Only score first N prompts per model")
    p.add_argument("--small-model", action="store_true",
                   help="Use small-model-aware prompt (31.7M context)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path (default: <results_dir>/judge_scores.json)")
    return p.parse_args()


def score_one(client: OpenAI, model: str, question: str, answer: str,
              max_retries: int, system_prompt: str = JUDGE_PROMPT) -> dict | None:
    answer_clean = answer.strip() if answer.strip() else "（模型未输出任何内容）"

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"问题：{question}\n回答：{answer_clean}"},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown code fences (```json ... ``` or ``` ... ```)
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
            if "{" in raw and "}" in raw:
                raw = raw[raw.find("{"):raw.rfind("}") + 1]
            scores = json.loads(raw)
            return scores
        except (json.JSONDecodeError, Exception) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [WARN] 解析失败（跳过）: {str(e)[:80]}")
                return None


def main():
    args = parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    active_prompt = JUDGE_PROMPT_SMALL_MODEL if args.small_model else JUDGE_PROMPT
    if args.small_model:
        print("[INFO] Using small-model-aware prompt (31.7M context)")

    scores_out = {}
    total_cost_est = 0

    for model_name, model_data in all_results.items():
        results = model_data["results"]
        if args.limit:
            results = results[:args.limit]
        model_scores = []
        dim_sums = {"相关性": 0.0, "流畅性": 0.0, "准确性": 0.0, "完整性": 0.0}
        count = 0

        print(f"\n{'='*50}")
        print(f"Scoring: {model_name} ({len(results)} prompts)")
        print(f"{'='*50}")

        for item in results:
            # Extract question from conversations
            conversations = item.get("input", [])
            if isinstance(conversations, list):
                question = next((c["content"] for c in conversations
                                 if c.get("role") == "user"), str(conversations))
            else:
                question = str(conversations)

            answer = item["output"]
            print(f"  [{item['id']}] {question[:50]}...", end=" ", flush=True)

            scores = score_one(client, args.model, question, answer, args.max_retries, active_prompt)
            if scores is None:
                continue
            avg = (scores.get("相关性", 0) + scores.get("流畅性", 0) +
                   scores.get("准确性", 0) + scores.get("完整性", 0)) / 4
            scores["_avg"] = round(avg, 2)
            scores["_id"] = item["id"]
            scores["_category"] = item["category"]
            scores["_question"] = question[:100]
            scores["_answer"] = answer[:200]
            model_scores.append(scores)

            print(f"avg={scores['_avg']:.1f}  note={scores.get('备注', '')[:40]}")
            count += 1
            for dim in dim_sums:
                dim_sums[dim] += scores.get(dim, 0)

            total_cost_est += 200  # ~200 tokens per response
            time.sleep(0.5)  # rate limit safety

        avg_row = {
            "相关性": round(dim_sums["相关性"] / count, 2) if count else 0,
            "流畅性": round(dim_sums["流畅性"] / count, 2) if count else 0,
            "准确性": round(dim_sums["准确性"] / count, 2) if count else 0,
            "完整性": round(dim_sums["完整性"] / count, 2) if count else 0,
        }
        avg_row["总分"] = round(sum(avg_row[d] for d in ["相关性", "流畅性", "准确性", "完整性"]) / 4, 2)

        scores_out[model_name] = {
            "checkpoint": model_data.get("checkpoint", ""),
            "per_prompt": model_scores,
            "averages": avg_row,
        }

        print(f"\n  {model_name} 平均分: {avg_row}")

    # ----- category breakdown -----
    for model_name, sd in scores_out.items():
        cats = {}
        for s in sd["per_prompt"]:
            cat = s.get("_category", "其他")
            cats.setdefault(cat, []).append(s["_avg"])
        sd["by_category"] = {c: round(sum(vs) / len(vs), 2) for c, vs in sorted(cats.items())}

    out_path = args.out or args.results.parent / "judge_scores.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scores_out, f, indent=2, ensure_ascii=False)

    # ----- summary table -----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'相关性':>6} {'流畅性':>6} {'准确性':>6} {'完整性':>6} {'总分':>6}")
    print("-" * 45)
    for mn, sd in scores_out.items():
        a = sd["averages"]
        print(f"{mn:<15} {a['相关性']:>6} {a['流畅性']:>6} {a['准确性']:>6} {a['完整性']:>6} {a['总分']:>6}")

    print(f"\nSaved to {out_path}")
    print(f"Estimated cost: ~${total_cost_est * 0.000002:.4f} (DeepSeek deepseek-chat)")


if __name__ == "__main__":
    main()
