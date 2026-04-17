"""
CDTP — KG & QA 任务评估脚本

评测指标：
    - Accuracy（完全匹配准确率）
    - MRR  (Mean Reciprocal Rank)
    - Hits@1 / Hits@3 / Hits@10
    - F1 Score（二分类）

使用方法：
    # 评测单个文件
    python kg_qa_eval.py --input results/kg_output.jsonl

    # 评测整个目录（默认）
    python kg_qa_eval.py --dir 1124/kg&qa

输出：
    - 终端打印评测结果
    - kg_qa_evaluation_results.xlsx
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

# -----------------------------------------------------------------------------
# 日志
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------


def clean_answer(answer: str) -> str:
    """清理答案字符串，去除选项标记。"""
    answer = answer.strip()
    answer = re.sub(r'^\s*[A-Z]、\s*', '', answer)
    answer = re.sub(r'^\d+、\s*', '', answer)
    return answer.strip()


def calculate_accuracy(filepath: str | Path) -> float:
    """
    计算完全匹配准确率（Accuracy）。

    文件格式：每行包含 'model_answer' 和 'standard_answer'。
    """
    correct = 0
    total = 0
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            model_ans = clean_answer(entry["model_answer"])
            std_ans = clean_answer(entry["standard_answer"])
            if model_ans == std_ans:
                correct += 1
            total += 1
    return (correct / total * 100) if total > 0 else 0.0


def read_queries_from_jsonl(filepath: str | Path) -> list[dict]:
    """
    从 JSONL 文件中读取查询，返回每个查询的正确排名。

    文件格式：每行包含 'model_answer_list' 和 'standard_answer'。
    """
    queries = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            model_list = [clean_answer(a) for a in data["model_answer_list"]]
            std_ans = clean_answer(data["standard_answer"])
            try:
                correct_rank = model_list.index(std_ans) + 1
            except ValueError:
                correct_rank = float("inf")
            queries.append({"query_id": data.get("query", ""), "correct_rank": correct_rank})
    return queries


def calculate_mrr(queries: list[dict]) -> float:
    """计算 MRR（Mean Reciprocal Rank）。"""
    if not queries:
        return 0.0
    rr_sum = sum(
        (1.0 / q["correct_rank"]) if q["correct_rank"] != float("inf") else 0.0
        for q in queries
    )
    return rr_sum / len(queries)


def calculate_hits_at_k(
    predictions: list[list[str]],
    correct_answers: list[str],
    k: int,
) -> float:
    """计算 Hits@K。"""
    hits = sum(
        1 for preds, correct in zip(predictions, correct_answers)
        if correct in preds[:k]
    )
    return (hits / len(correct_answers) * 100) if correct_answers else 0.0


def load_predictions_and_labels(filepath: str | Path) -> tuple:
    """
    加载预测列表和标准答案列表。

    Returns:
        predictions, correct_answers, true_labels, predicted_labels
    """
    predictions: list[list[str]] = []
    correct_answers: list[str] = []
    true_labels: list[int] = []
    predicted_labels: list[int] = []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            preds = [clean_answer(a) for a in entry["model_answer_list"]]
            std_ans = clean_answer(entry["standard_answer"])
            predictions.append(preds)
            correct_answers.append(std_ans)
            true_labels.append(1)
            predicted_labels.append(1 if preds[0] == std_ans else 0)

    return predictions, correct_answers, true_labels, predicted_labels


# -----------------------------------------------------------------------------
# 主评测函数
# -----------------------------------------------------------------------------


def evaluate_file(filepath: str | Path) -> dict:
    """评测单个文件，返回所有指标。"""
    filepath = Path(filepath)

    accuracy = calculate_accuracy(filepath)
    queries = read_queries_from_jsonl(filepath)
    preds, correct, true_lbl, pred_lbl = load_predictions_and_labels(filepath)

    mrr = calculate_mrr(queries)
    hits_at_1 = calculate_hits_at_k(preds, correct, 1)
    hits_at_3 = calculate_hits_at_k(preds, correct, 3)
    hits_at_10 = calculate_hits_at_k(preds, correct, 10)
    f1 = f1_score(true_lbl, pred_lbl, average="binary")

    return {
        "filename": filepath.name,
        "MRR": round(mrr, 4),
        "Accuracy": round(accuracy / 100, 4),
        "Hits@1": round(hits_at_1 / 100, 4),
        "Hits@3": round(hits_at_3 / 100, 4),
        "Hits@10": round(hits_at_10 / 100, 4),
        "F1": round(f1, 4),
    }


def evaluate_directory(folder: str | Path, pattern: str = "*.jsonl") -> list[dict]:
    """评测目录中所有 JSONL 文件。"""
    folder = Path(folder)
    results = []
    for filepath in sorted(folder.glob(pattern)):
        if filepath.is_file():
            logger.info("评测文件：%s", filepath.name)
            try:
                res = evaluate_file(filepath)
                results.append(res)
                _print_result(res)
            except Exception as exc:
                logger.error("评测失败 [%s]：%s", filepath.name, exc)
    return results


def _print_result(res: dict) -> None:
    """格式化打印评测结果。"""
    print(f"\n{'='*50}")
    print(f"📄 文件：{res['filename']}")
    print(f"{'='*50}")
    print(f"  MRR       : {res['MRR']:.4f}")
    print(f"  Accuracy  : {res['Accuracy']:.2%}")
    print(f"  Hits@1    : {res['Hits@1']:.2%}")
    print(f"  Hits@3    : {res['Hits@3']:.2%}")
    print(f"  Hits@10   : {res['Hits@10']:.2%}")
    print(f"  F1 Score  : {res['F1']:.4f}")
    print("-" * 50)


# -----------------------------------------------------------------------------
# 入口
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="CDTP — KG & QA Evaluation")
    parser.add_argument("--dir", default="1124/kg&qa", help="JSONL 文件所在目录")
    parser.add_argument("--input", dest="single_file", default=None, help="评测单个文件")
    parser.add_argument("--output", default="kg_qa_evaluation_results.xlsx", help="Excel 输出路径")
    args = parser.parse_args()

    if args.single_file:
        # 评测单个文件
        results = [evaluate_file(args.single_file)]
        _print_result(results[0])
    else:
        # 评测目录
        results = evaluate_directory(args.dir)

    if results:
        df = pd.DataFrame(results)
        df.to_excel(args.output, index=False)
        print(f"\n✅ 结果已保存至：{args.output}")
    else:
        print("⚠️  未找到可评测的文件。")
        sys.exit(1)


if __name__ == "__main__":
    main()
