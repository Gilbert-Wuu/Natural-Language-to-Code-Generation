import argparse
import multiprocessing
import os

import torch
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging,
    set_seed,
)
from trl import SFTTrainer, SFTConfig
import wandb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="bigcode/starcoder2-3b")
    parser.add_argument("--dataset_name", type=str, default="the-stack-smol")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset_text_field", type=str, default="content")

    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.set_defaults(bf16=True)
    
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="finetune_starcoder2")
    parser.add_argument("--num_proc", type=int, default=None)
    
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--merge", action="store_true")

    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")

    parser.add_argument("--wandb_project", type=str, default="starcoder2-finetuning", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity (username or team)")

    # Evaluation arguments
    parser.add_argument("--eval_split", type=str, default="test", help="Split to use for evaluation")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation during training")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint frequency")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep")

    return parser.parse_args()


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main(args):
    # config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    # load model and tokenizer
    token = os.environ.get("HF_TOKEN", None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": PartialState().process_index},
        attention_dropout=args.attention_dropout,
        token=token,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    
    print_trainable_parameters(model)

    # load data
    train_data = load_dataset(
        args.dataset_name,
        args.subset,
        split=args.split,
        token=token,
        num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
    )
    
    eval_data = None
    if args.do_eval:
        try:
            eval_data = load_dataset(
                args.dataset_name,
                args.subset,
                split=args.eval_split,
                token=token,
                num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
            )
            print(f"Loaded {len(eval_data)} examples for evaluation")
        except Exception as e:
            print(f"Warning: Could not load eval split '{args.eval_split}': {e}")
            eval_data = None

    # data preprocess
    if "CodeAlpaca" in args.dataset_name:
        def format_codealpaca(example):
            # Standard Alpaca template
            prompt_template = (
                "### Instruction:\n"
                "{instruction}\n\n"
                "### Output:\n"
            )
            
            # Format the prompt
            instruction = example['prompt']
            response = example['completion']
            text = prompt_template.format(instruction=instruction) + response

            return {"text": text}
        
        train_data = train_data.map(format_codealpaca, remove_columns=train_data.column_names)
        if eval_data is not None:
            eval_data = eval_data.map(format_codealpaca, remove_columns=eval_data.column_names)
        args.dataset_text_field = "text"
        print("Formatted CodeAlpaca dataset")
        print("Sample:", train_data[0]["text"][:200])

    # setup training
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps" if args.do_eval and eval_data is not None else "no",
        eval_steps=args.eval_steps if args.do_eval else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True if args.do_eval and eval_data is not None else False,
        metric_for_best_model="eval_loss" if args.do_eval else None,
        optim="paged_adamw_8bit",
        seed=args.seed,
        report_to="wandb" if args.use_wandb else "none",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # SFT-specific parameters
        dataset_text_field=args.dataset_text_field,
        packing=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data if args.do_eval else None,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # launch
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),  # Log all hyperparameters
        )
    print("Training...")
    trainer.train()

    print("Saving the last checkpoint of the model")
    trainer.save_model(os.path.join(args.output_dir, "final_checkpoint"))
    
    if args.push_to_hub:
        if args.merge:
            print("Merging LoRA adapters with base model...")
            
            # Clean up training model to free memory
            del model
            del trainer
            torch.cuda.empty_cache()
            
            print("Loading base model in bf16...")
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=token,
            )
            
            print("Loading LoRA adapters...")
            peft_model = PeftModel.from_pretrained(
                base_model,
                os.path.join(args.output_dir, "final_checkpoint"),
                torch_dtype=torch.bfloat16,
            )
            
            print("Merging...")
            merged_model = peft_model.merge_and_unload()
            
            # Extract base name from output_dir to avoid namespace issues
            base_name = os.path.basename(args.output_dir.rstrip('/'))
            repo_name = f"{base_name}-merged"
            
            # Save locally first to ensure clean state
            temp_save_path = os.path.join(args.output_dir, "merged_model")
            print(f"Saving merged model to {temp_save_path}...")
            
            # Get state dict and save without PEFT attributes
            state_dict = merged_model.state_dict()
            
            # Clean up PEFT model references
            del peft_model
            del base_model
            torch.cuda.empty_cache()
            
            # Reload as clean transformers model
            print("Reloading as clean transformers model...")
            clean_model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=token,
            )
            
            # Load merged weights
            clean_model.load_state_dict(state_dict, strict=False)
            
            # Clean up temporary objects
            del merged_model
            del state_dict
            torch.cuda.empty_cache()
            
            # Save the clean model
            clean_model.save_pretrained(temp_save_path)
            tokenizer.save_pretrained(temp_save_path)
            
            print(f"Pushing to hub as {repo_name}...")
            clean_model.push_to_hub(
                repo_id=repo_name,
                commit_message="Upload merged model",
                token=token,
            )
            tokenizer.push_to_hub(
                repo_id=repo_name,
                commit_message="Upload tokenizer",
                token=token,
            )
            
            print(f"✓ Model uploaded successfully to: https://huggingface.co/{repo_name}")
        else:
            trainer.push_to_hub("Upload model")
            
    if args.use_wandb:
        wandb.finish()

    print("Training Done! 💥")


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)