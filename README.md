# CDTP

> **Bridging Structured Knowledge and Text: The CDTP Dataset for Evaluating Chinese Large Language Models**
>
> TKDE 2026 / ICML 2026 Submission

---

## Overview

CDTP is a benchmark for evaluating Chinese LLMs on structured knowledge understanding and text generation. It covers three tasks across four domains:

| Task | Description | Metrics |
|------|-------------|---------|
| **KG** | Knowledge graph completion | MRR, Hits@K, Accuracy |
| **QA** | Multi-choice question answering | MRR, Hits@K, Accuracy |
| **TextGen** | Structured-to-text generation | BLEU, ROUGE, METEOR |

**Domains:** History & Politics · Humanities & Society · Nature & Environment · Technology & Economics
**Models:** Yi-9B · Qwen1.5-7B · GLM-4-9B · DeepSeek-7B · Llama-3-8B · Baichuan2-7B · InternLM2-7B · Phi-2

---

## Quick Start

```bash
# 1. Environment
conda env create -f environment.yml && conda activate cdtp-repro

# 2. Base Model inference
cd BaseModel/History_and_Politics
python combined_tasks_History_and_Politics_yi_9b.py

# 3. SFT Model inference
cd ../../SFTModel/History_and_Politics/kg
python knowledge_map_completion_yi_9b.py

# 4. Evaluate
python Evaluation/kg_qa_eval.py --dir 1124/kg&qa
python Evaluation/textgen_eval.py --dir 1124/textgen
```

---

## Directory Structure

```
CDTP-main/
├── cdtp_templates.py          # Shared prompt templates
├── BaseModel/                 # Base Model inference (multi-process, all tasks)
│   ├── History_and_Politics/
│   ├── Humanities_and_Society/
│   ├── Nature_and_Environment/
│   └── Technology_and_Economics/
│       └── combined_tasks_<Domain>_<model>.py
│
├── SFTModel/                  # SFT Model inference (per-task scripts)
│   ├── History_and_Politics/
│   │   ├── kg/                # knowledge_map_completion_<model>.py
│   │   ├── qa/                # qa_<model>.py
│   │   └── textgen/           # text_gen_<model>.py
│   ├── Humanities_and_Society/
│   └── Technology_and_Economics/
│
├── Evaluation/                # Evaluation pipeline
│   ├── kg_qa_eval.py          # KG & QA metrics (MRR, Hits@K, F1)
│   ├── textgen_eval.py        # TextGen metrics (BLEU, ROUGE, METEOR)
│   ├── clean_data/
│   │   └── clean_all.py       # Unified cleaner: --task qa|kg|textgen
│   └── transform_data/
│       ├── kmc_transform.py
│       ├── qa_transform.py
│       └── textgen_transform.py
│
└── Test/                       # Quick test scripts (Yi-9B)
    ├── BaseModel/
    └── SFTModel/
```

---

## Data Pipeline

```
Raw JSONL → clean_all.py → transform_data/*.py → Evaluation/*.py
```

| Step | Command |
|------|---------|
| Clean KG/QA | `python Evaluation/clean_data/clean_all.py --task kg --input_folder X --output_folder Y` |
| Clean TextGen | `python Evaluation/clean_data/clean_all.py --task textgen --input_folder X --output_folder Y` |
| Eval KG/QA | `python Evaluation/kg_qa_eval.py --dir transformed/kg_qa` |
| Eval TextGen | `python Evaluation/textgen_eval.py --dir transformed/textgen` |

---

## Environment

```yaml
name: cdtp-repro
dependencies:
  - python=3.10
  - pip
  - pip:
      - torch>=2.6.0
      - transformers>=4.35
      - deepspeed==0.17.5
      - accelerate
      - safetensors
      - datasets
      - scikit-learn
      - sacrebleu
      - nltk
      - rouge-score
      - jieba
      - pandas
      - numpy
```

---

## Training (DeepSpeed ZeRO-3 + CPU Offload)

```bash
deepspeed --num_gpus 8 run_finetune.py \
  --deepspeed \
  --deepspeed_config configs/deepspeed_zero3_offload.json \
  --exp_name qwen3_32b_textgen_History_and_Politics \
  --data_path data/CDTP \
  --model_name_or_path /path/to/Qwen3-32B \
  --num_train_epochs 3 \
  --zero_stage 3 \
  --offload \
  --use_bf16 \
  --output_dir checkpoints/qwen3_32b_textgen_History_and_Politics
```

Resume: add `--resume_from_checkpoint checkpoints/epoch-1-xxxx-zero`

---

## Checkpoint Conversion (ZeRO → HuggingFace)

```bash
# Option A — DeepSpeed utility
python deepspeed/utils/zero_to_fp32.py \
  --checkpoint_dir checkpoints/epoch-2-2655-zero \
  --output_file merged/pytorch_model.bin

# Option B — Custom helper
python scripts/convert_zero_to_hf_simple.py \
  /path/to/epoch-2-2655-zero \
  /path/to/epoch-2-2655-hf

cp /path/to/Qwen3-32B/tokenizer* /path/to/epoch-2-2655-hf/
```

---

## Hardware

| Setting | Spec |
|---------|------|
| Primary | 8 × A800 (80 GB) |
| Minimum | Single GPU |

---

## Citation

```bibtex
@article{cdtp2026,
  title  = {Bridging Structured Knowledge and Text: The CDTP Dataset
            for Evaluating Chinese Large Language Models},
  author = {Jiapu Wang et al.},
  journal = {TKDE},
  year    = {2026},
  note    = {Under review — ICML 2026 submission}
}
```

---

*Last updated: 2025*
