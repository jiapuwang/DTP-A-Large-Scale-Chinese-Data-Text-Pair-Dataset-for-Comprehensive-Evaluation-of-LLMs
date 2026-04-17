"""
CDTP — Text Generation 任务评估脚本

评测指标：
    - BLEU-1 / BLEU-2 / BLEU-3 / BLEU-4
    - ROUGE-1 / ROUGE-2 / ROUGE-L
    - METEOR

使用方法：
    # 评测目录（默认）
    python textgen_eval.py --dir 1124/textgen

    # 评测单个文件
    python textgen_eval.py --input results/textgen_output.jsonl

输出：
    - 终端打印各项分数
    - textgen_results.xlsx
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import jieba
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge

# -----------------------------------------------------------------------------
# 日志
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

sys.setrecursionlimit(5000)

# -----------------------------------------------------------------------------
# 评分函数
# -----------------------------------------------------------------------------


def calculate_bleu(predicted: str, reference: str) -> tuple[float, float, float, float]:
    """
    计算 BLEU-1 ~ BLEU-4。

    Returns:
        (bleu1, bleu2, bleu3, bleu4)
    """
    pred_tokens = list(jieba.cut(predicted))
    ref_tokens = [list(jieba.cut(reference))]

    scores = []
    for i in range(1, 5):
        weights = tuple(1.0 if j == i - 1 else 0.0 for j in range(4))
        score = sentence_bleu(ref_tokens, pred_tokens, weights=weights)
        scores.append(score)
    return tuple(scores)  # type: ignore[return-value]


def calculate_rouge(predicted: str, reference: str) -> dict:
    """计算 ROUGE-1 / ROUGE-2 / ROUGE-L。"""
    rouge = Rouge()
    pred_segmented = " ".join(jieba.cut(predicted))
    ref_segmented = " ".join(jieba.cut(reference))
    scores = rouge.get_scores(pred_segmented, ref_segmented, avg=True)
    return {
        "rouge-1": scores["rouge-1"]["f"],
        "rouge-2": scores["rouge-2"]["f"],
        "rouge-l": scores["rouge-l"]["f"],
    }


def calculate_meteor(predicted: str, reference: str) -> float:
    """计算 METEOR 分数。"""
    pred_tokens = list(jieba.cut(predicted))
    ref_tokens = list(jieba.cut(reference))
    return single_meteor_score(pred_tokens, ref_tokens)


# -----------------------------------------------------------------------------
# 数据集评估
# -----------------------------------------------------------------------------


def evaluate_file(
    filepath: str | Path,
    max_length: int = 1000,
) -> Optional[dict]:
    """
    评测单个 JSONL 文件。

    每行格式：
        {"model_answer": "...", "standard_answer": "..."}
    """
    filepath = Path(filepath)

    bleu_scores: list[tuple[float, float, float, float]] = []
    rouge_scores: list[dict] = []
    meteor_scores: list[float] = []

    skipped = 0

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            model_answer = data.get("model_answer", "").strip()
            std_answer = data.get("standard_answer", "").strip()

            # 长度过滤
            if not model_answer or not std_answer:
                logger.warning("跳过空答案行")
                skipped += 1
                continue

            if len(model_answer) > max_length or len(std_answer) > max_length:
                logger.warning("跳过超长答案（model=%d, std=%d）",
                               len(model_answer), len(std_answer))
                skipped += 1
                continue

            try:
                bleu = calculate_bleu(model_answer, std_answer)
                rouge = calculate_rouge(model_answer, std_answer)
                meteor = calculate_meteor(model_answer, std_answer)
                bleu_scores.append(bleu)
                rouge_scores.append(rouge)
                meteor_scores.append(meteor)
            except Exception as exc:
                logger.error("计算分数失败：%s", exc)
                skipped += 1

    if not bleu_scores:
        logger.error("文件无可用评测数据：%s", filepath.name)
        return None

    total = len(bleu_scores)
    avg_bleu = [sum(s[i] for s in bleu_scores) / total for i in range(4)]
    avg_rouge = {
        "rouge-1": sum(s["rouge-1"] for s in rouge_scores) / total,
        "rouge-2": sum(s["rouge-2"] for s in rouge_scores) / total,
        "rouge-l": sum(s["rouge-l"] for s in rouge_scores) / total,
    }
    avg_meteor = sum(meteor_scores) / total

    return {
        "filename": filepath.name,
        "total_samples": total,
        "skipped": skipped,
        "BLEU-1": round(avg_bleu[0], 4),
        "BLEU-2": round(avg_bleu[1], 4),
        "BLEU-3": round(avg_bleu[2], 4),
        "BLEU-4": round(avg_bleu[3], 4),
        "ROUGE-1": round(avg_rouge["rouge-1"], 4),
        "ROUGE-2": round(avg_rouge["rouge-2"], 4),
        "ROUGE-L": round(avg_rouge["rouge-l"], 4),
        "METEOR": round(avg_meteor, 4),
    }


def evaluate_directory(folder: str | Path, pattern: str = "*.jsonl") -> list[dict]:
    """评测目录下所有 JSONL 文件。"""
    folder = Path(folder)
    results = []
    for filepath in sorted(folder.glob(pattern)):
        if filepath.is_file():
            logger.info("评测文件：%s", filepath.name)
            res = evaluate_file(filepath)
            if res is not None:
                results.append(res)
                _print_result(res)
    return results


def _print_result(res: dict) -> None:
    """格式化打印评测结果。"""
    print(f"\n{'='*55}")
    print(f"📄 文件：{res['filename']}  (有效样本：{res['total_samples']}，跳过：{res['skipped']}）")
    print(f"{'='*55}")
    print(f"  BLEU-1  : {res['BLEU-1']:.4f}")
    print(f"  BLEU-2  : {res['BLEU-2']:.4f}")
    print(f"  BLEU-3  : {res['BLEU-3']:.4f}")
    print(f"  BLEU-4  : {res['BLEU-4']:.4f}")
    print(f"  ROUGE-1 : {res['ROUGE-1']:.4f}")
    print(f"  ROUGE-2 : {res['ROUGE-2']:.4f}")
    print(f"  ROUGE-L : {res['ROUGE-L']:.4f}")
    print(f"  METEOR  : {res['METEOR']:.4f}")
    print("-" * 55)


# -----------------------------------------------------------------------------
# 入口
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="CDTP — Text Generation Evaluation")
    parser.add_argument("--dir", default="1124/text_gen", help="JSONL 文件所在目录")
    parser.add_argument("--input", dest="single_file", default=None, help="评测单个文件")
    parser.add_argument("--output", default="textgen_results.xlsx", help="Excel 输出路径")
    parser.add_argument("--max_length", type=int, default=1000, help="答案最大字符数阈值")
    args = parser.parse_args()

    if args.single_file:
        results = [evaluate_file(args.single_file, args.max_length)]
        results = [r for r in results if r is not None]
        if results:
            _print_result(results[0])
    else:
        results = evaluate_directory(args.dir)

    if results:
        df = pd.DataFrame(results)
        # 排序列名顺序
        cols = ["filename", "total_samples", "skipped",
                "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
                "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR"]
        df = df[[c for c in cols if c in df.columns]]
        df.to_excel(args.output, index=False)
        print(f"\n✅ 结果已保存至：{args.output}")
    else:
        print("⚠️  未找到可评测的文件。")
        sys.exit(1)


if __name__ == "__main__":
    main()
