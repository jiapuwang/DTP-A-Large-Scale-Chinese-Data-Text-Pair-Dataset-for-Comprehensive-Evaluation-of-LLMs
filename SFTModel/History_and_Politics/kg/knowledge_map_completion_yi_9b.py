"""
CDFT — SFT 模型推理：知识图谱补全（Knowledge Map Completion）

单任务脚本，一对一对应，无硬编码路径冲突。

使用方法：
    # 直接运行（修改下方 CONFIG 即可）
    python knowledge_map_completion_yi_9b.py

    # 或通过命令行参数覆盖
    python knowledge_map_completion_yi_9b.py \
        --model_path /path/to/model \
        --checkpoint /path/to/checkpoint.pt \
        --input_file /path/to/input.jsonl \
        --output_file /path/to/output.jsonl \
        --gpu 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# ⚙️ 配置区 — 修改这里即可适配不同模型/路径
# =============================================================================

CONFIG = {
    # 模型路径（基础模型路径，SFT checkpoint 另行指定）
    "model_path": "/public/chengweiwu/models/cdtp/Yi-9B",

    # SFT checkpoint 路径（如无需加载 checkpoint，设为 None）
    "checkpoint": "/public/chengweiwu/mingyang/sft/checkpoints/Yi-9B_qa_History_and_Politics/epoch-2-252-hf/pytorch_model.bin",

    # GPU 编号
    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),

    # 日志路径
    "log_dir": Path("/public/chengweiwu/mingyang/logs"),
    "log_name": "kmc_History_and_Politics_yi_9b.log",

    # 数据路径
    "input_file": "/public/chengweiwu/mingyang/data/input_data/submit/qa_submit/qa_all_History_and_Politics_random_1w1_head1w_dev2k.jsonl",
    "output_file": "/public/chengweiwu/mingyang/data/output_data/kmc_output_History_and_Politics_yi_9b.jsonl",

    # 生成参数
    "max_length": 2500,
    "do_sample": True,
    "top_k": 1,
}

# =============================================================================
# Few-shot 示例（与 base 脚本保持一致）
# =============================================================================

_KG_EXAMPLES = (
    "知识图谱是由三元组构成的，其中三元组形式为 (头实体,关系,尾实体)。 知识图谱补全是对缺失三元组推断出缺失的信息。以下是一些需要补全的实例，以及对应的十个选项，"
    "你是知识图谱补全的专家，请给出?处缺失的信息，并根据你认为的十个选项的准确程度给出排序，答案简洁，无需做任何解释，答案以<SOD>开始<EOD>结束。"
    "实例：(赖上被上帝遗忘的天使,作者,?) ?处缺失的信息：A、芈夏, B、芈秋, C、芈冬, D、芈春, E、芈梅, F、芈兰, G、芈菊, H、芈荷, I、芈雪, J、芈露，答案：<SOD>F、芈兰, B、芈秋, A、芈夏, E、芈梅, C、芈冬, J、芈露, D、芈春, H、芈荷, I、芈雪, G、芈菊<EOD>"
    "实例：(老字号网店,主办,?) ?处缺失的信息：A、上海老字号协会, B、北京新字号协会, C、北京老字号协会, D、上海新字号协会, E、南京老字号协会, F、南京新字号协会, G、广州新字号协会, H、广州老字号协会, I、西安新字号协会, J、西安老字号协会，答案：<SOD>C、上海老字号协会, I、西安新字号协会, D、上海新字号协会, F、南京新字号协会, B、北京新字号协会, E、南京老字号协会, G、广州新字号协会, H、广州老字号协会, A、北京老字号协会, J、西安老字号协会<EOD>"
    "实例：(泉水寨,总面积,?) ?处缺失的信息：A、39平方公里, B、29平方公里, C、19平方公里, D、9平方公里, E、49平方公里, F、59平方公里, G、69平方公里, H、79平方公里, I、89平方公里, J、99平方公里，答案：<SOD>C、29平方公里, A、9平方公里, G、69平方公里, D、39平方公里, F、59平方公里, E、49平方公里, H、79平方公里, B、19平方公里, I、89平方公里, J、99平方公里<EOD>"
    "实例：(林乐伦,出生地,?) ?处缺失的信息：A、湖北衡东县, B、湖南衡东县, C、河北衡东县, D、河南衡东县, E、山东衡东县, F、山西衡东县, G、广东衡东县, H、广西衡东县, I、江西衡东县, J、江苏衡东县，答案：<SOD>I、江西衡东县, D、河南衡东县, C、湖北衡东县, F、山西衡东县, A、湖南衡东县, E、山东衡东县, G、广东衡东县, B、湖北衡东县, H、广西衡东县, J、江苏衡东县<EOD>"
)


def build_prompt(data: dict) -> str:
    """构建 KG 补全提示词。"""
    kmc = data["post"].get("knowledge_map_completion", "")
    return _KG_EXAMPLES + kmc + " ?处缺失的信息：" + data["response"] + "，答案："


# =============================================================================
# 日志
# =============================================================================

def setup_logging(log_dir: Path, log_name: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_dir / log_name),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


# =============================================================================
# 推理
# =============================================================================

def run_inference(
    model_path: str,
    checkpoint: Optional[str],
    input_file: str,
    output_file: str,
    gpu: str,
    max_length: int,
    do_sample: bool,
    top_k: int,
) -> float:
    """
    执行 KG 推理并返回耗时（秒）。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载基础模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )

    # 加载 SFT checkpoint（如指定）
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        logging.info("✓ Checkpoint 已加载：%s", checkpoint)

    model.eval()
    start_time = time.time()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    skipped = 0

    with open(input_file, encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line)
            query = build_prompt(data)

            if not query.strip():
                logging.warning("Query 为空，跳过。")
                skipped += 1
                continue

            inputs = tokenizer(query, return_tensors="pt").to(device)
            gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_k": top_k}

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

            if outputs.size(0) == 0:
                logging.warning("模型生成空输出，跳过。")
                skipped += 1
                continue

            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            data["model_answer"] = answer
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            outfile.flush()

    elapsed = time.time() - start_time
    logging.info("✓ 完成（%.1fs，跳过 %d 条）：%s", elapsed, skipped, output_file)
    return elapsed


# =============================================================================
# 入口
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="CDTP — SFT KG Completion Inference")
    for key in CONFIG:
        parser.add_argument(f"--{key}", default=CONFIG[key])
    args = parser.parse_args()

    # 日志
    log_dir = Path(args.log_dir)
    log_name = args.log_name if hasattr(args, "log_name") else CONFIG["log_name"]
    setup_logging(log_dir, log_name)

    logging.info("=" * 60)
    logging.info("🔧 CDTP SFT KG Completion Inference")
    logging.info("   Model     : %s", args.model_path)
    logging.info("   Checkpoint: %s", args.checkpoint)
    logging.info("   GPU       : %s", args.gpu)
    logging.info("=" * 60)

    elapsed = run_inference(
        model_path=args.model_path,
        checkpoint=args.checkpoint,
        input_file=args.input_file,
        output_file=args.output_file,
        gpu=args.gpu,
        max_length=int(args.max_length),
        do_sample=bool(args.do_sample),
        top_k=int(args.top_k),
    )
    print(f"\n✅ 推理完成，耗时 {elapsed:.1f}s")


if __name__ == "__main__":
    main()
