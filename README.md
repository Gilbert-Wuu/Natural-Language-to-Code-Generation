# Natural-Language-to-Code-Generation: In-Context Learning vs. Parameter-Efficient Fine-Tuning

## 🚀 Project Overview
This repository documents a comparative study between **In-Context Learning (ICL)** and **Parameter-Efficient Fine-Tuning (PEFT)**, specifically using LoRA, for instruction-to-code generation tasks. We investigate how these two prominent tuning methodologies affect the performance of small-scale Code Large Language Models (LLMs).

Our experiments focus on the **Qwen2.5-Coder-3B** and **Deepseek-Coder-1.3B** model families, trained on the CodeAlpaca-20k dataset, and evaluated across various code generation benchmarks.

---

## 📁 Repository Structure

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

---

## 🛠️ Methodology and Core Components

### 1. Models and Dataset

| Component | Details | Role in Experiment |
| :--- | :--- | :--- |
| **LLMs Used** | Qwen2.5-Coder-3B-Instruct & Deepseek-Coder-1.3B-Instruct | Primarily for **ICL** and as baselines |
| **SFT Models** | Qwen2.5-Coder-3B (Base) & Deepseek-Coder-1.3B (Base) | Used for **LoRA Fine-Tuning** |
| **Dataset** | CodeAlpaca-20k (~20,000 instruction-code pairs) | Used for Supervised Fine-Tuning (SFT). Covers Python, JS, Java, C++ |

### 2. Experiment Methods

| Method | Description | Implementation Details | Key Notebooks |
| :--- | :--- | :--- | :--- |
| **ICL (In-Context Learning)** | Few-shot prompting without updating model weights. | **Prompt selection** on MBPP (100 problems) to find optimal shot count (0, 1, 3, 5). Best prompt applied to full benchmarks. | `DS-ICL.ipynb`, `Eval_ICL_Qwen.ipynb`, `Eval_ICL_Qwen_Large_Model.ipynb` |
| **SFT (Supervised Fine-Tuning)** | PEFT using **LoRA** on base models. | LoRA rank sweep (r=16/32/64) for 500 steps, followed by full 1,500-step training with best config. | `SFT-Train.ipynb` |

### 3. Evaluation

* **Metric:** Pass@k (primarily Pass@1) — functional correctness via code execution
* **Benchmarks:**
  * **HumanEval** — 164 Python problems
  * **HumanEval+** — extended HumanEval with stricter test cases
  * **InstructHumanEval** — instruction-following variant of HumanEval
  * **MultiPL-E** — cross-language: C++, Java, JavaScript
  * **MBPP** — used for ICL prompt selection (Phase 1)
* **Evaluation framework:** [`bigcode-evaluation-harness`](https://github.com/bigcode-project/bigcode-evaluation-harness)

---

## 🔧 LoRA Fine-Tuning Details

Training is handled by `finetune.py`, which wraps HuggingFace `peft` + `trl`'s `SFTTrainer`.

### LoRA Rank Sweep (Phase 1 — 500 steps each)

| Config | `lora_r` | `lora_alpha` | Selected For |
| :--- | :--- | :--- | :--- |
| High rank | 64 | 128 | DeepSeek (best config) |
| Mid rank | 32 | 64 | Qwen (best config) |
| Minimal rank | 16 | 32 | — |

### Training Hyperparameters

| Parameter | Phase 1 (Sweep) | Phase 2 (Full Training) |
| :--- | :--- | :--- |
| `max_steps` | 500 | 1500 |
| `learning_rate` | 2e-4 | 2e-4 |
| `lr_scheduler_type` | cosine | cosine |
| `warmup_steps` | 50 | 300 |
| `micro_batch_size` | 4 | 4 |
| `gradient_accumulation_steps` | 4 (effective batch=16) | 4 (effective batch=16) |
| `weight_decay` | 0.05 | 0.05 |
| `lora_dropout` | 0.1 | 0.1 |
| `attention_dropout` | 0.1 | 0.1 |

**Other settings:** 4-bit NF4 quantization, `bfloat16`, `paged_adamw_8bit`, target modules: `q/k/v/o_proj`, `gate/up/down_proj`

---

## 💡 Key Findings

### A. Model Sensitivity to In-Context Learning

ICL effectiveness is highly model-dependent, requiring careful prompt selection:

* **Qwen2.5-Coder-3B-Instruct:** Showed a **monotonic improvement** as the number of shots increased, peaking at **5-shot**. The model effectively leverages provided examples.
* **Deepseek-Coder-1.3B-Instruct:** Performed best with **0-shot**. Adding few-shot examples degraded performance, suggesting the model treats extra context as noise.

### B. Effects of LoRA Fine-Tuning (SFT)

| Model | HumanEval Pass@1 (Pre-SFT) | HumanEval Pass@1 (Post-SFT) | Finding |
| :--- | :--- | :--- | :--- |
| **Deepseek-Coder-1.3B** | ~44% (Instruct baseline) | **Improved** | SFT successfully adapted the model, with performance gains across Python benchmarks. |
| **Qwen2.5-Coder-3B** | ~52% (Base) | ~26% | SFT caused **severe degradation**, likely due to overfitting to CodeAlpaca-20k's format, which diverges from HumanEval's style. |

---

## ⚙️ How to Reproduce Experiments

**Prerequisites:** Python 3.x, CUDA-capable GPU (experiments run on Google Colab L4), Hugging Face Hub access.

### 1. Install Dependencies

```bash
git clone https://github.com/Gilbert-Wuu/Natural-Language-to-Code-Generation.git
cd Natural-Language-to-Code-Generation
pip install -r requirements.txt
```

### 2. Run LoRA Fine-Tuning

```bash
# Example: fine-tune Qwen 3B with mid-rank LoRA config
python finetune.py \
    --model_id "Qwen/Qwen2.5-Coder-3B" \
    --dataset_name "HuggingFaceH4/CodeAlpaca_20K" \
    --subset "default" \
    --split "train" \
    --max_steps 1500 \
    --micro_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 300 \
    --weight_decay 0.05 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --attention_dropout 0.1 \
    --seed 11667 \
    --output_dir "checkpoints/qwen-lora-r32-a64-full" \
    --use_wandb \
    --merge \
    --push_to_hub
```

### 3. Run Notebooks

Execute notebooks in the `notebooks/` directory in the following order:

| Step | Notebook(s) | Purpose |
| :--- | :--- | :--- |
| **1. ICL Prompt Selection** | `Eval_ICL_Qwen.ipynb`, `DS-ICL.ipynb` | Sweep 0/1/3/5-shot on MBPP to find the best prompt per model. |
| **2. SFT Training** | `SFT-Train.ipynb` | LoRA rank sweep + full training. Runs on Colab (L4 GPU). |
| **3. Final Evaluation** | `Eval_ICL_Qwen.ipynb`, `DS-ICL.ipynb`, `SFT-Eval.ipynb` | Full benchmarks (HumanEval, HumanEval+, MultiPL-E) for both ICL and SFT. |

---

Contributors: Arthur Chien, Yaxuan Mao, Gilbert Wu
