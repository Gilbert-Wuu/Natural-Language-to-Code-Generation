# Natural-Language-to-Code-Generation: In-Context Learning vs. Parameter-Efficient Fine-Tuning

## 🚀 Project Overview
This repository documents a comparative study between **In-Context Learning (ICL)** and **Parameter-Efficient Fine-Tuning (PEFT)**, specifically using LoRA, for instruction-to-code generation tasks. We investigate how these two prominent tuning methodologies affect the performance of small-scale Code Large Language Models (LLMs).

Our experiments focus on the **Qwen2.5-Coder-3B** and **Deepseek-Coder-1.3B** model families, trained on the CodeAlpaca-20k dataset, and evaluated across various code generation benchmarks.

## 🛠️ Methodology and Core Components

### 1. Models and Dataset

| Component | Details | Role in Experiment |
| :--- | :--- | :--- |
| **LLMs Used** | Qwen2.5-Coder-3B-Instruct & Deepseek-Coder-1.3B-Instruct | Primarily for **ICL** and as baselines. |
| **SFT Models** | Qwen2.5-Coder-3B (Base) & Deepseek-Coder-1.3B (Base) | Used for **LoRA Fine-Tuning**. |
| **Dataset** | CodeAlpaca-20k (~20,000 instruction-code pairs) | Used for Supervised Fine-Tuning (SFT). Covers Python, JS, Java, C++. |

### 2. Experiment Methods

| Method | Description | Implementation Details | Key Notebooks |
| :--- | :--- | :--- | :--- |
| **ICL (In-Context Learning)** | Using Few-shot examples via prompting without updating weights. | **Prompt Selection** performed on MBPP to determine the optimal number of shots (0, 1, 3, 5). | `DS-ICL.ipynb`, `Eval_ICL_Qwen.ipynb`, `Eval_ICL_Qwen_Large_Model.ipynb` |
| **SFT (Supervised Fine-Tuning)** | PEFT using **LoRA** on the base models. | Hyperparameter sweep for LoRA parameters (`r`, `alpha`), followed by a full 1,500-step training. | `SFT-Train.ipynb` |

### 3. Evaluation

* **Metric:** Pass@k (primarily Pass@1) to measure functional correctness.
* **Benchmarks:** HumanEval, HumanEval+, InstructHumanEval (Python) and MultiPL-E (Cross-Language: C++, Java, JavaScript).

## 💡 Key Findings

### A. Model Sensitivity to In-Context Learning

ICL effectiveness is highly model-dependent, requiring careful prompt selection:

* **Qwen2.5-Coder-3B-Instruct:** Showed a **monotonic improvement** in performance as the number of shots increased, peaking at **5-shot**. This indicates the model effectively learns patterns from the provided examples.
* **Deepseek-Coder-1.3B-Instruct:** Performed best with the **Baseline (0-shot)** prompt. Adding few-shot examples resulted in a **performance decrease**, suggesting the model might treat the extra context as noise or distraction.

### B. Effects of LoRA Fine-Tuning (SFT)

| Model | HumanEval Pass@1 (Pre-SFT) | HumanEval Pass@1 (Post-SFT) | Finding |
| :--- | :--- | :--- | :--- |
| **Deepseek-Coder-1.3B** | ~44% (Instruct) | **Improved** | SFT successfully adapted the model to the instruction-following format, leading to performance gains across Python benchmarks. |
| **Qwen2.5-Coder-3B** | ~52% (Base) | ~26% (SFT) | SFT caused **severe performance degradation** on standard benchmarks, likely due to **overfitting** to the specific format of the CodeAlpaca-20k training dataset. |

## ⚙️ How to Reproduce Experiments

All experiments utilized the `bigcode-evaluation-harness` for evaluation and an external LoRA fine-tuning script (`finetune.py`, referenced in `SFT-Train.ipynb`).

### 1. Environment Setup

**Prerequisites:** Python 3.x, access to a GPU, and Hugging Face Hub access.

1.  **Clone Dependencies & Install:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Authenticate Hugging Face:**
    ```python
    from huggingface_hub import notebook_login
    notebook_login()
    ```

### 2. Running the Notebooks

Execute the relevant Jupyter Notebooks located in the `notebooks/` directory following the execution order:

| Step | Notebook(s) | Purpose |
| :--- | :--- | :--- |
| **1. ICL Prompt Selection** | `Eval_ICL_Qwen.ipynb`, `DS-ICL.ipynb` | Determine the optimal few-shot prefix for each model on MBPP. |
| **2. SFT Training** | `SFT-Train.ipynb` | Perform LoRA hyperparameter search and final SFT training runs. **Remember to update model paths in the notebook.** |
| **3. Final Evaluation** | `Eval_ICL_Qwen.ipynb`, `DS-ICL.ipynb`, `SFT-Eval.ipynb` | Run final benchmarks (HumanEval, MultiPL-E) for the best ICL prompts and the SFT checkpoints. |

---
Contributors: Arthur Chien, Yaxuan Mao, Gilbert Wu
