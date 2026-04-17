"""
Base Model Combined Inference — All domains & all models
在基础模型上并行执行 KG / QA / Text Gen 三任务推理。

使用方法：
    python combined_tasks_History_and_Politics_yi_9b.py

依赖：
    pip install torch transformers
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.multiprocessing import set_start_method
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# 本地模板（当 cdtp_templates.py 不在同目录时使用）
# -----------------------------------------------------------------------------


class _Templates:
    """内置提示词模板（作为 cdtp_templates.py 的备用）。"""

    QA_EXAMPLES = (
        "请回答下边的问题，要求答案根据选项的准确程度给出排序，答案简洁，无需做任何解释，答案以<SOD>开始<EOD>结束。"
        "问题：(±)反式菊酸的分子式是什么？A、C9H16O2, B、C10H16O2, C、C11H16O2, D、C11H16O3, E、C12H18O2, F、C10H15O2, G、C10H16O3, H、C9H15O2, I、C11H15O2, J、C10H17O2，答案：<SOD>B、C10H16O2, J、C10H17O2, D、C11H16O3, F、C10H15O2, I、C11H15O2, A、C9H16O2, H、C9H15O2, C、C11H16O2, E、C12H18O2, G、C10H16O3<EOD>"
        "问题：11·15马鲁古海地震的地震级数是几级？A、6.2, B、8.9, C、7.2, D、3.5, E、7.5, F、6.8, G、7.0, H、5.9, I、8.2, J、7.3，答案：<SOD>B、8.9, F、6.8, H、5.9, C、7.2, D、3.5, I、8.2, G、7.0, A、6.2, E、7.5, J、7.3<EOD>"
        "问题：破山击出自哪部作品？A、海贼王, B、龙珠, C、破坏王, D、火影忍者, E、死神, F、全职猎人, G、妖精的尾巴, H、进击的巨人, I、银魂, J、勇者斗恶龙，答案：<SOD>C、破坏王, F、全职猎人, H、进击的巨人, B、龙珠, I、银魂, E、死神, A、海贼王, J、勇者斗恶龙, D、火影忍者, G、妖精的尾巴<EOD>"
        "问题：龙芯2号的研制单位是哪个？A、中国科学研究院自动化所, B、中国科学研究院软件所, C、中国科学研究院计算技术研究所, D、中国科学研究院化工所, E、中国社科院, F、清华大学, G、北京大学, H、复旦大学, I、上海交通大学, J、浙江大学，答案：<SOD>C、中国科学研究院计算技术研究所, B、中国科学研究院软件所, D、中国科学研究院化工所, A、中国科学研究院自动化所, E、中国社科院, F、清华大学, J、浙江大学, G、北京大学, I、上海交通大学, H、复旦大学<EOD>"
    )

    TEXT_GEN_EXAMPLES = (
        "三元组是知识图谱中的一种结构化表示形式，用于描述实体之间的关系。一个三元组由三个元素组成：[(头实体,关系,尾实体)]。"
        "目标文本生成旨在根据输入的结构化三元组生成与三元组对应的自然语言文本。"
        "你是文本生成领域的专家，请基于输入三元组：[(头实体,关系,尾实体)]生成目标文本，"
        "以下是一些需要生成目标文本的实例，要求生成的目标文本简洁，只需给出生成的目标文本，无需做任何解释，生成的目标文本以<SOD>开始<EOD>结束。"
        "实例：三元组：[(9·11国家纪念博物馆,组成,国家纪念馆和国家博物馆),(9·11国家纪念博物馆,地理位置,世贸中心遗址),(9·11国家纪念博物馆,目的,纪念在9·11事件中失去生命的美国人民)]"
        "生成的文本：<SOD>9·11国家纪念博物馆是为了纪念在9·11事件中失去生命的美国人民，它分为国家纪念馆和国家博物馆两个部分，位于纽约\"9·11\"恐怖袭击的发生地——世贸中心遗址。<EOD>"
        "实例：三元组：[(Biography\"John Travolta,导演，Adam Friedman),(Biography\"John Travolta,主演,Halle Berry)]"
        "生成的文本：<SOD>《\"Biography\"John Travolta》是由Adam Friedman执导，Halle Berry、Nicolas Cage主演的一部影片。<EOD>"
        "实例：三元组：[(范例表示,外文名,case representation),(范例表示,所属学科,信息科学技术),(范例表示,公布年度,2008年)]"
        "生成的文本：<SOD>范例表示（case representation）是2008年公布的信息科学技术名词。<EOD>"
        "实例：三元组：[(矩磁材料,归属,磁性材料)]"
        "生成的文本：<SOD>矩磁材料是具有矩形磁滞回线、剩余磁感强度Br和工作时最大磁感应强度Bm的比值,即Br/Bm接近于1以及矫顽磁力较小的磁性材料。<EOD>"
        "实例：三元组：[(一种困惑感,导演,克劳德·朱特),(一种困惑感,编剧,克劳德·朱特、Clément Perron),(一种困惑感,主演,Jacques Gagnon、Lyne Champagne)]"
        "生成的文本：<SOD>《一种困惑感》是由克劳德·朱特执导的电影，由克劳德·朱特、Clément Perron担任编剧，Jacques Gagnon、Lyne Champagne等主演。<EOD>"
    )

    KG_EXAMPLES = (
        "知识图谱是由三元组构成的，其中三元组形式为 (头实体,关系,尾实体)。 知识图谱补全是对缺失三元组推断出缺失的信息。以下是一些需要补全的实例，以及对应的十个选项，"
        "你是知识图谱补全的专家，请给出?处缺失的信息，并根据你认为的十个选项的准确程度给出排序，答案简洁，无需做任何解释，答案以<SOD>开始<EOD>结束。"
        "实例：(赖上被上帝遗忘的天使,作者,?) ?处缺失的信息：A、芈夏, B、芈秋, C、芈冬, D、芈春, E、芈梅, F、芈兰, G、芈菊, H、芈荷, I、芈雪, J、芈露，答案：<SOD>F、芈兰, B、芈秋, A、芈夏, E、芈梅, C、芈冬, J、芈露, D、芈春, H、芈荷, I、芈雪, G、芈菊<EOD>"
        "实例：(老字号网店,主办,?) ?处缺失的信息：A、上海老字号协会, B、北京新字号协会, C、北京老字号协会, D、上海新字号协会, E、南京老字号协会, F、南京新字号协会, G、广州新字号协会, H、广州老字号协会, I、西安新字号协会, J、西安老字号协会，答案：<SOD>C、上海老字号协会, I、西安新字号协会, D、上海新字号协会, F、南京新字号协会, B、北京新字号协会, E、南京老字号协会, G、广州新字号协会, H、广州老字号协会, A、北京老字号协会, J、西安老字号协会<EOD>"
        "实例：(泉水寨,总面积,?) ?处缺失的信息：A、39平方公里, B、29平方公里, C、19平方公里, D、9平方公里, E、49平方公里, F、59平方公里, G、69平方公里, H、79平方公里, I、89平方公里, J、99平方公里，答案：<SOD>C、29平方公里, A、9平方公里, G、69平方公里, D、39平方公里, F、59平方公里, E、49平方公里, H、79平方公里, B、19平方公里, I、89平方公里, J、99平方公里<EOD>"
        "实例：(林乐伦,出生地,?) ?处缺失的信息：A、湖北衡东县, B、湖南衡东县, C、河北衡东县, D、河南衡东县, E、山东衡东县, F、山西衡东县, G、广东衡东县, H、广西衡东县, I、江西衡东县, J、江苏衡东县，答案：<SOD>I、江西衡东县, D、河南衡东县, C、湖北衡东县, F、山西衡东县, A、湖南衡东县, E、山东衡东县, G、广东衡东县, B、湖北衡东县, H、广西衡东县, J、江苏衡东县<EOD>"
    )

    @staticmethod
    def qa_template(data: dict) -> str:
        return _Templates.QA_EXAMPLES + "问题：" + data["response"] + "，答案："

    @staticmethod
    def text_gen_template(data: dict) -> str:
        triple_str = ""
        for element in data.get("triples", []):
            parts = element.split("\t")
            if len(parts) < 3:
                raise ValueError(f"triples 字段格式错误，期望 3 个字段，实际：{parts}")
            triple_str += f"({parts[0]},{parts[1]},{parts[2]}),"
        triple_str = re.sub(r'\)\s*,$', ')', triple_str)
        return _Templates.TEXT_GEN_EXAMPLES + triple_str + "] 生成的文本："

    @staticmethod
    def kg_template(data: dict) -> str:
        kmc = data["post"].get("knowledge_map_completion", "")
        return _Templates.KG_EXAMPLES + kmc + " ?处缺失的信息：" + data["response"] + "，答案："


# -----------------------------------------------------------------------------
# 任务配置
# -----------------------------------------------------------------------------


@dataclass
class TaskConfig:
    """单个推理任务配置。"""
    input_file: str
    output_file: str
    template_fn: Callable[[dict], str]

    @classmethod
    def from_dict(cls, data: dict) -> "TaskConfig":
        return cls(
            input_file=data["input_file"],
            output_file=data["output_file"],
            template_fn=data["template"],
        )


from dataclasses import dataclass

# -----------------------------------------------------------------------------
# 默认配置（修改此处即可适配不同模型）
# -----------------------------------------------------------------------------

# ⚙️ 修改 MODEL_PATH 适配不同模型
MODEL_PATH = "/public/chengweiwu/models/cdtp/Yi-9B"

# ⚙️ 修改 GPU 编号（如需多卡）
GPU_IDS = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS

# ⚙️ 生成参数
GEN_KWARGS = {
    "max_length": 2500,
    "do_sample": True,
    "top_k": 1,
}

# ⚙️ 日志路径
LOG_DIR = Path("/public/chengweiwu/mingyang/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"combined_tasks_yi_9b.log"


def configure_logging(log_file: Optional[str | Path] = None) -> None:
    """配置日志记录器。"""
    if log_file is None:
        log_file = LOG_FILE
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


# -----------------------------------------------------------------------------
# 模型加载
# -----------------------------------------------------------------------------


def load_model(model_path: str) -> tuple:
    """
    加载模型和分词器。

    Returns:
        (tokenizer, model, device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    ).eval()
    return tokenizer, model, device


# -----------------------------------------------------------------------------
# 推理函数
# -----------------------------------------------------------------------------


def process_single_entry(
    entry: dict,
    model,
    tokenizer,
    device: str,
    template_fn: Callable[[dict], str],
    gen_kwargs: dict,
) -> Optional[dict]:
    """处理单条数据并返回结果。"""
    query = template_fn(entry)
    if not query.strip():
        logging.error("Query 为空，跳过该条。")
        return None

    inputs = tokenizer(query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    if outputs.size(0) == 0:
        logging.error("模型生成空输出，query: %s", query[:100])
        return None

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info("生成长度 %d 字符", len(answer))
    entry["model_answer"] = answer
    return entry


def process_task(
    task: TaskConfig,
    model,
    tokenizer,
    device: str,
) -> None:
    """
    处理单个任务文件。
    """
    logging.info("▶ 开始任务：%s → %s", task.input_file, task.output_file)
    start_time = time.time()

    Path(task.output_file).parent.mkdir(parents=True, exist_ok=True)
    skipped = 0

    with open(task.input_file, "r", encoding="utf-8") as infile, \
         open(task.output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            entry = json.loads(line)
            result = process_single_entry(
                entry, model, tokenizer, device,
                task.template_fn, GEN_KWARGS,
            )
            if result is not None:
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                outfile.flush()
            else:
                skipped += 1

    elapsed = time.time() - start_time
    logging.info("✓ 任务完成（%.1fs，跳过 %d 条）：%s", elapsed, skipped, task.output_file)


# -----------------------------------------------------------------------------
# 入口
# -----------------------------------------------------------------------------


def get_default_tasks(
    model_name: str = "yi_9b",
    output_root: str = "/public/chengweiwu/mingyang/data/output_data/1115",
    input_root: str = "/public/chengweiwu/mingyang/data/input_data/submit",
) -> list[TaskConfig]:
    """
    返回所有域的 Text Generation 任务配置（可按需修改为 KG / QA）。
    适配不同模型时请重写此函数。
    """
    domains = [
        "History_and_Politics",
        "Humanities_and_Society",
        "Nature_and_Environment",
        "Technology_and_Economics",
    ]
    tasks = []
    for domain in domains:
        task = TaskConfig(
            input_file=(
                f"{input_root}/textGen_submit/"
                f"textGen_all_{domain}_random_1w1_head1w_dev2k.jsonl"
            ),
            output_file=f"{output_root}/text_gen_output_{domain}_{model_name}.jsonl",
            template_fn=_Templates.text_gen_template,
        )
        tasks.append(task)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="CDTP Base Model Combined Inference")
    parser.add_argument("--model_path", default=MODEL_PATH, help="模型路径")
    parser.add_argument("--log_file", default=str(LOG_FILE), help="日志文件路径")
    parser.add_argument("--num_workers", type=int, default=4, help="并行进程数")
    args = parser.parse_args()

    configure_logging(args.log_file)
    logging.info("=" * 60)
    logging.info("🔧 CDTP Base Model Inference")
    logging.info("   Model: %s", args.model_path)
    logging.info("   GPU  : %s", GPU_IDS)
    logging.info("=" * 60)

    # 加载模型（全局一份，避免重复加载）
    tokenizer, model, device = load_model(args.model_path)

    # 获取任务列表（默认执行所有域的 Text Gen）
    tasks = get_default_tasks()

    # 多进程并行
    with Pool(processes=args.num_workers) as pool:
        results = []
        for task in tasks:
            results.append(
                pool.apply_async(
                    process_task,
                    (task, model, tokenizer, device),
                )
            )
        for res in results:
            try:
                res.get()
            except Exception as exc:
                logging.error("任务异常：%s", exc)

    logging.info("🎉 所有任务完成。")


if __name__ == "__main__":
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
