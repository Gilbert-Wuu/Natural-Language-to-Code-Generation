# Natural Language to Code Generation

## Project Overview

This project investigates two approaches for instruction-to-code generation using small coding language models:

1. **In-Context Learning (ICL)**: Few-shot prompting without any weight updates
2. **Supervised Fine-Tuning (SFT) with LoRA**: Parameter-efficient fine-tuning using Low-Rank Adaptation

The goal is to compare how well each approach enables models to generate correct Python (and multi-language) code from natural language instructions.

---

## Models

| Model | Size | Used For |
|---|---|---|
| `Qwen/Qwen2.5-Coder-3B-Instruct` | 3B | ICL evaluation |
| `Qwen/Qwen2.5-Coder-3B` | 3B | LoRA fine-tuning |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | ICL evaluation (extended experiment) |
| `deepseek-ai/deepseek-coder-1.3b-instruct` | 1.3B | ICL evaluation |
| `deepseek-ai/deepseek-coder-1.3b-base` | 1.3B | LoRA fine-tuning |

---

## Training Dataset

**CodeAlpaca-20K** (`HuggingFaceH4/CodeAlpaca_20K`)
- ~20,000 instruction-code pairs
- Format: `prompt` (instruction) + `completion` (code)
- Preprocessed into Alpaca template:
  ```
  ### Instruction:
  {instruction}

  ### Output:
  {code}
  ```
- HuggingFace dataset split: `train` for training, `test` for evaluation

---

## LoRA Fine-Tuning (`finetune.py`)

### Framework
- **PEFT**: `LoraConfig` from HuggingFace `peft`
- **Trainer**: `SFTTrainer` from `trl`
- **Quantization**: 4-bit NF4 (`BitsAndBytesConfig`) for memory efficiency
- **Mixed precision**: `bfloat16`
- **Optimizer**: `paged_adamw_8bit`

### LoRA Configuration

| Parameter | Value |
|---|---|
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Task type | `CAUSAL_LM` |
| LoRA dropout | 0.1 |

### Hyperparameter Sweep (Phase 1 — Config Tuning, 500 steps)

Three LoRA rank configurations were tested:

| Config | `lora_r` | `lora_alpha` | Notes |
|---|---|---|---|
| Config 1 (High rank) | 64 | 128 | alpha = 2× rank |
| Config 2 (Mid rank) | 32 | 64 | alpha = 2× rank |
| Config 3 (Minimal rank) | 16 | 32 | alpha = 2× rank |

### Shared Training Hyperparameters

| Parameter | Exploration (Phase 1) | Full Training (Phase 2) |
|---|---|---|
| `max_steps` | 500 | 1500 |
| `micro_batch_size` | 4 | 4 |
| `gradient_accumulation_steps` | 4 | 4 |
| Effective batch size | 16 | 16 |
| `learning_rate` | 2e-4 | 2e-4 |
| `lr_scheduler_type` | cosine | cosine |
| `warmup_steps` | 50 | 300 |
| `weight_decay` | 0.05 | 0.05 |
| `attention_dropout` | 0.1 | 0.1 |
| `seed` | 11667 | 11667 |
| `eval_steps` | 50 | 50 |
| `save_steps` | 250 | 250 |
| `save_total_limit` | 3 | 3 |

### Full Training (Phase 2)
- **Qwen**: Config 2 (r=32, alpha=64) selected as best
- **DeepSeek**: Config 1 (r=64, alpha=128) selected as best
- After training: LoRA adapters merged into base model weights and pushed to HuggingFace Hub

### Model Checkpoints (HuggingFace Hub)
- `arthur-chien-0530/11667-qwen-lora-r32-a64-full-merged`

### Experiment Tracking
- **Weights & Biases** (`wandb`), project: `11667-experiments`

---

## In-Context Learning (ICL)

### Strategy
- **Phase 1**: Prompt selection on MBPP (first 100 problems) — sweep 0-shot, 1-shot, 3-shot, 5-shot
- **Phase 2**: Evaluate best prompt on full benchmarks (HumanEval, HumanEval+, MultiPL-E)

### Few-Shot Prompt Format
Pure Python code examples (no Markdown, no comments outside function body) with a hard rule prefix:
```
# You are a Python coding assistant.
# Only output valid Python code implementing the required function.
# Do NOT use markdown or ```.
# Do NOT print explanations or comments outside the function body.
```

Examples used: `factorial`, `is_palindrome`, `fibonacci`, `find_max`, `reverse_list`

### Generation Parameters

| Parameter | Value |
|---|---|
| `temperature` | 0.2 |
| `top_p` | 0.95 |
| `max_new_tokens` | 2048 |
| `do_sample` | True |
| `n_samples` | 10 (for pass@k) |

### ICL Results Summary

| Model | Best Shot Config | Reason |
|---|---|---|
| Qwen2.5-Coder-3B-Instruct | 5-shot | Monotonic improvement with more shots |
| DeepSeek-Coder-1.3B-Instruct | 0-shot | Performance degrades with additional shots |

---

## Evaluation Benchmarks

### 1. HumanEval
- **164** Python programming problems
- Metric: **Pass@1** (functional correctness via code execution)
- Library: `human-eval` (`from human_eval.data import read_problems`)

### 2. HumanEval+
- Extended version of HumanEval with stricter/more test cases
- Metric: **Pass@1**, **Pass@10**
- Library: `evalplus` (`from evalplus.data import get_human_eval_plus`)

### 3. MBPP (Mostly Basic Programming Problems)
- Google Research dataset: `google-research-datasets/mbpp`
- Used **first 100 problems** (`split="test"`) for prompt selection (Phase 1)
- Metric: Pass@1

### 4. MultiPL-E
- Cross-language code generation benchmark
- Languages evaluated: **C++**, **Java**, **JavaScript**
- Run via bigcode-evaluation-harness

### 5. InstructHumanEval
- Instruction-following variant of HumanEval
- Evaluated in `Eval_ICL_Qwen.ipynb`

---

## bigcode-evaluation-harness

Used for standardized benchmark evaluation, especially MultiPL-E.

- **Repo**: `https://github.com/arthur900530/bigcode-evaluation-harness` (fork of `bigcode/bigcode-evaluation-harness`)
- **Installation**: `pip install -e .`

### Example Usage

```bash
# HumanEval+ evaluation with 5-shot prefix
PREFIX="$(cat prompts/mbpp_5shot.txt)"
python main.py \
  --model "Qwen/Qwen2.5-Coder-3B-Instruct" \
  --tasks "humanevalplus" \
  --top_p 0.95 \
  --temperature 0.2 \
  --do_sample True \
  --n_samples 10 \
  --batch_size 10 \
  --max_length 2048 \
  --prefix_code "$PREFIX"

# MultiPL-E C++ evaluation
python main.py \
  --model "Qwen/Qwen2.5-Coder-3B-Instruct" \
  --tasks "multiple-cpp" \
  --top_p 0.95 \
  --temperature 0.2 \
  --do_sample True \
  --n_samples 10 \
  --batch_size 10 \
  --max_length 2048
```

---

## Repository Structure

```
Natural-Language-to-Code-Generation/
├── finetune.py                          # LoRA SFT training script (main entry point)
├── requirements.txt                     # Python dependencies
├── notebooks/
│   ├── SFT-Train.ipynb                  # LoRA hyperparameter sweep + full training
│   ├── SFT-Eval.ipynb                   # Post-SFT benchmark evaluation
│   ├── Eval_ICL_Qwen.ipynb              # ICL: Qwen 3B — prompt selection + full eval
│   ├── DS-ICL.ipynb                     # ICL: DeepSeek 1.3B — prompt exploration
│   └── Eval_ICL_Qwen_Large_Model.ipynb  # ICL: Qwen 7B — extended model size experiment
└── reports/
    ├── 11_667_Homework_6_Report.pdf
    └── Instruction-to-Code...pptx
```

### Key Files

- **`finetune.py`**: Accepts CLI arguments for model, dataset, LoRA config, and training hyperparameters. Supports `--push_to_hub` and `--merge` flags to merge LoRA adapters and upload to HuggingFace Hub.
- **`SFT-Train.ipynb`**: Calls `finetune.py` with different LoRA rank configs. Clones this repo on Colab and runs training on L4 GPU.
- **`Eval_ICL_Qwen.ipynb`**: Full ICL pipeline for Qwen 3B — MBPP prompt selection, HumanEval, HumanEval+, MultiPL-E (C++/Java/JS), InstructHumanEval.
- **`Eval_ICL_Qwen_Large_Model.ipynb`**: Lightweight ICL eval for larger Qwen models (7B/15B) using the best 5-shot prompt. Self-contained eval loop (no bigcode-harness) since prompt is dynamically built per problem.
- **`DS-ICL.ipynb`**: ICL exploration for DeepSeek 1.3B.

---

## Key Findings

| Approach | Model | Result |
|---|---|---|
| ICL | Qwen2.5-Coder-3B-Instruct | 5-shot is best; monotonic improvement with more shots |
| ICL | DeepSeek-Coder-1.3B-Instruct | 0-shot is best; additional shots hurt performance |
| LoRA SFT | Qwen2.5-Coder-3B | Severe overfitting: Pass@1 dropped from ~52% → ~26%. Likely caused by format mismatch between CodeAlpaca and HumanEval |
| LoRA SFT | DeepSeek-Coder-1.3B | Successful fine-tuning with performance improvement on HumanEval |

---

## Environment

- **Runtime**: Google Colab (GPU: NVIDIA L4)
- **Python**: 3.12
- **Key dependencies**: `transformers>=4.46`, `peft>=0.13`, `trl>=0.11`, `accelerate>=1.1`, `bitsandbytes>=0.44`, `datasets>=3.0`, `wandb>=0.18`
