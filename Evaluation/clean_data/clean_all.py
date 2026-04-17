"""
CDTP — 数据清洗工具（统一版）

支持 KG / QA / Text Gen 三种任务的数据清洗，统一入口。

用法：
    # QA / KG 数据清洗（提取 post, response, model_answer）
    python clean_all.py --task qa \
        --input_folder 1124/qa \
        --output_folder cleaned/1124/qa

    # Text Gen 数据清洗（提取 entity, text, triples, matched_triples, model_answer）
    python clean_all.py --task text_gen \
        --input_folder 1124/text_gen \
        --output_folder cleaned/1124/text_gen

    # KG 数据清洗（与 qa 完全相同逻辑）
    python clean_all.py --task kg \
        --input_folder 1124/kg \
        --output_folder cleaned/1124/kg
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

# -----------------------------------------------------------------------------
# 日志
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 答案提取
# -----------------------------------------------------------------------------

def extract_answer(raw_answer: str) -> str | None:
    """
    从模型原始输出中提取答案。

    支持格式：
      1. <SOD>...<EOD>
      2. 答案：...<EOD>
      3. ...<EOD>（纯 EOD 兜底）
    """
    answer = raw_answer

    if "<SOD>" in answer and "<EOD>" in answer:
        return answer.split("<SOD>")[1].split("<EOD>")[0].strip()

    if "答案：" in answer and "<EOD>" in answer:
        idx = answer.find("答案：") + len("答案：")
        return answer[idx:].split("<EOD>")[0].strip()

    if "<EOD>" in answer:
        return answer.split("<EOD>")[0].strip()

    logger.warning("无法解析答案格式：%s", raw_answer[:80])
    return None


# -----------------------------------------------------------------------------
# 各任务数据提取
# -----------------------------------------------------------------------------

def clean_qa_kg_entry(data: dict) -> dict | None:
    """
    QA / KG 任务的数据提取。

    输入：{"post": {...}, "response": "...", "model_answer": "..."}
    输出：{"post": {...}, "response": "...", "model_answer": "..."}
    """
    post = data.get("post", {})
    response = data.get("response", "")
    model_answer = extract_answer(data.get("model_answer", ""))
    if model_answer is None:
        return None
    return {"post": post, "response": response, "model_answer": model_answer}


def clean_text_gen_entry(data: dict) -> dict | None:
    """
    Text Gen 任务的数据提取。

    输入：{"entity": "", "text": "", "triples": [], "matched_triples": [], "model_answer": "..."}
    输出：同上（只保留必要字段 + model_answer）
    """
    model_answer = data.get("model_answer", "")
    # text_gen 只用 <SOD>...<EOD> 格式
    if "<SOD>" in model_answer and "<EOD>" in model_answer:
        model_answer = model_answer.split("<SOD>")[1].split("<EOD>")[0].strip()
    else:
        logger.warning("无法解析 text_gen 答案格式：%s", model_answer[:80])
        return None

    return {
        "entity": data.get("entity", ""),
        "text": data.get("text", ""),
        "triples": data.get("triples", []),
        "matched_triples": data.get("matched_triples", []),
        "model_answer": model_answer,
    }


# -----------------------------------------------------------------------------
# 核心清洗
# -----------------------------------------------------------------------------

CLEANERS = {
    "qa": clean_qa_kg_entry,
    "kg": clean_qa_kg_entry,
    "text_gen": clean_text_gen_entry,
}


def clean_file(input_path: str | Path, output_path: str | Path, task: str) -> tuple[int, int]:
    """
    清洗单个 JSONL 文件。

    Returns:
        (total_lines, kept_lines)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cleaner = CLEANERS[task]
    total = 0
    kept = 0

    with open(input_path, encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            total += 1
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.error("JSON 解析失败 [%s:%d]：%s", input_path.name, total, exc)
                continue

            cleaned = cleaner(data)
            if cleaned is None:
                continue

            outfile.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept += 1

    return total, kept


def batch_clean(input_folder: str, output_folder: str, task: str) -> None:
    """批量清洗目录中的所有 JSONL 文件。"""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(input_folder.glob("*.jsonl"))
    if not files:
        logger.warning("目录中未找到 .jsonl 文件：%s", input_folder)
        return

    logger.info("发现 %d 个文件 [%s]，开始清洗...", len(files), task)
    grand_total = 0
    grand_kept = 0

    for filepath in files:
        total, kept = clean_file(filepath, output_folder / filepath.name, task)
        grand_total += total
        grand_kept += kept
        rate = (kept / total * 100) if total > 0 else 0
        logger.info("✓ %s  — 保留 %d/%d (%.1f%%)", filepath.name, kept, total, rate)

    logger.info("=" * 50)
    logger.info("🎉 完成！总计 %d 条，保留 %d 条（%.1f%%）",
                 grand_total, grand_kept,
                 (grand_kept / grand_total * 100) if grand_total > 0 else 0)


# -----------------------------------------------------------------------------
# 入口
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CDTP — 统一数据清洗工具")
    parser.add_argument("--task", choices=["qa", "kg", "text_gen"], required=True,
                        help="任务类型：qa / kg / text_gen")
    parser.add_argument("--input_folder", required=True, help="原始 JSONL 目录")
    parser.add_argument("--output_folder", required=True, help="清洗后 JSONL 目录")
    args = parser.parse_args()

    logger.info("📂 输入：%s", args.input_folder)
    logger.info("📂 输出：%s", args.output_folder)
    logger.info("📋 任务：%s", args.task)

    batch_clean(args.input_folder, args.output_folder, args.task)
    print("\n✅ 数据清洗完成！")


if __name__ == "__main__":
    main()
