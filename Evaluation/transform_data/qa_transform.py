"""
CDTP — QA 数据格式转换

将清洗后的 QA 数据转换为评测格式。

输入格式：
    {"post": {"Tuples3": "..."},
     "response": "A、..., B、..., ...",
     "model_answer": "A、..., B、..., ..."}

输出格式：
    {"model_answer_list": [...], "model_answer": "...",
     "query": "...", "standard_answer": "..."}

用法：
    python qa_transform.py \
        --input_folder /path/to/cleaned/qa \
        --output_folder /path/to/transformed/kg_qa
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def transform_file(input_path: str | Path, output_path: str | Path) -> int:
    """转换单个文件，返回处理行数。"""
    count = 0
    with open(input_path, encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line.strip())
            model_answer_list = data["model_answer"].split(", ")
            model_answer = model_answer_list[0]  # 第一个为最优答案
            query = data["response"]
            standard_answer = data["post"]["Tuples3"]

            converted = {
                "model_answer_list": model_answer_list,
                "model_answer": model_answer,
                "query": query,
                "standard_answer": standard_answer,
            }
            outfile.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="CDTP — QA 数据格式转换")
    parser.add_argument("--input_folder",
                        default="/share/project/chengweiwu/code/CB-ECLLM/haipeng/code/eval/cleaned/qa",
                        help="清洗后 QA 数据目录")
    parser.add_argument("--output_folder",
                        default="/share/project/chengweiwu/code/CB-ECLLM/haipeng/code/eval/transformed/kg_qa",
                        help="转换后数据输出目录")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    logger.info("输入目录：%s", args.input_folder)
    logger.info("输出目录：%s", args.output_folder)

    files = sorted(Path(args.input_folder).glob("*.jsonl"))
    if not files:
        logger.warning("目录中未找到 .jsonl 文件：%s", args.input_folder)
        return

    total = 0
    for filepath in files:
        out = os.path.join(args.output_folder, filepath.name)
        count = transform_file(filepath, out)
        total += count
        logger.info("✓ %s（%d 条）→ %s", filepath.name, count, out)

    logger.info("🎉 完成，共处理 %d 条数据。", total)


if __name__ == "__main__":
    main()
