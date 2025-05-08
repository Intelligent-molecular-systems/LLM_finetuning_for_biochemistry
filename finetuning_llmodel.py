"""
This script finetunes a HuggingFace CausalLM (Llama3) model using LoRA adapters
with 4-bit quantization support and WandB integration.
"""

import os
import json
import argparse
import torch
import wandb
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer


def formatting_func(example):
    """ Function to format the input conversation (system, user and assistant) for the LLM training """
    
    output_texts = [] 
    # Begin each full conversation with <|begin_of_text|>
    for i in range(len(example['message'])):
        text = (f"<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n\n"
                f"{example['message'][i][0]['content']}<|eot_id|>"
                
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{example['message'][i][1]['content']}<|eot_id|>"
                
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{example['message'][i][2]['content']}<|eot_id|>")

        output_texts.append(text)
    
    return output_texts



def load_tokenizer_and_model(quantize, model_id, HF_TOKEN):
    
    print("\n########## Hugging Face path:", model_id)    
    print("\n########## Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                            token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    print("\n########## Loading base model...\n")
    if quantize == True:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            )

        model = AutoModelForCausalLM.from_pretrained(
                model_id,
                # device_map='auto',
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                token=HF_TOKEN,
                low_cpu_mem_usage=True 
            )
    elif quantize == False:

        model = AutoModelForCausalLM.from_pretrained(
                model_id,
                # device_map='auto',
                torch_dtype=torch.bfloat16,
                token=HF_TOKEN,
                low_cpu_mem_usage=True
            )

    print("\n\n########## 4-bit quantization: ", quantize)
    return tokenizer, model



def set_config(target_modules, Lora_directory, r, alpha, lr, scheduler):

    print("\n\n########## Preparing configuration files for LoRA finetuning\n")
    
    peft_config = LoraConfig(
        r=r,                  
        lora_alpha=alpha,     
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="lora_only",
        task_type="CAUSAL_LM"
    )

    sft_config = SFTConfig(
        max_seq_length=512,
        output_dir=Lora_directory,
        auto_find_batch_size=True,
        learning_rate=lr,
        lr_scheduler_type=scheduler,        # default is linear
        num_train_epochs=15,                
        logging_dir="./logs",               
        eval_strategy="epoch",              # Evaluate at the end of each epoch
        logging_strategy="epoch",           # Log at the end of each epoch
        save_strategy="epoch",              # Save model at the end of each epoch
        save_total_limit=1,                 # Keep only the best checkpoint
        report_to="wandb",                  # Log metrics to Weights & Biases
        load_best_model_at_end=True,        # Load the best model at the end
        gradient_checkpointing=True,        # Enable the computations of gradients on the fly   
    )

    wandb_config = {
        'learning_rate': lr,
        'rank': r,
        'lora_alpha': peft_config.lora_alpha,
        'lora_dropout': peft_config.lora_dropout,
        'lora_bias': peft_config.bias,
        'epochs': sft_config.num_train_epochs,
    }

    return peft_config, sft_config, wandb_config



def train(tokenizer, model, sft_config, peft_config, wandb_config, dataset, model_name, resume_from_checkpoint=None):

    # size = int(600) # to train on lower data regimes 
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset['train'],     # .select(range(size)),
        eval_dataset=dataset['validation'], # .select(range(size)),
        peft_config=peft_config,
        formatting_func=formatting_func,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Initialize WandB based on whether we are resuming or starting fresh
    if resume_from_checkpoint:
    
        run_id = '' # Insert a valid ID here to resume a run
        print(f"\n########## Resuming training from checkpoint: {resume_from_checkpoint}\n")
        wandb.init(
            project="LLM_PoC_finetunings",
            name=model_name,
            config=wandb_config,
            resume="allow",
            id=run_id
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        print("\n########## Starting training from scratch\n")
        wandb.init(
            project="LLM_PoC_finetunings",
            name=model_name,
            config=wandb_config
        )
        trainer.train()

    wandb.finish()



def main():

    if "HF_HOME" in os.environ:
        hf_home_path = os.environ["HF_HOME"]
        print(f"HF_HOME is set to: {hf_home_path}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Base model for finetuning",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        choices=["meta-llama/Meta-Llama-3.1-70B-Instruct", 
                 "meta-llama/Meta-Llama-3.1-8B-Instruct"],
    )
    parser.add_argument(
        "--dataset_path",
        default='./dataset.hf',
    )
    parser.add_argument(
        "--lora_directory",
        default="./Finetuning_default",
    )
    parser.add_argument(
        "--rank",
        default=16,
    )
    parser.add_argument(
        "--lora_alpha",
        default=32,
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-4,
    )
    parser.add_argument(
        "--scheduler",
        default="linear",
    )
    parser.add_argument(
        "--model_name",
        default="prova",
    )
    parser.add_argument(
        "--lora_type",
        default="light",
        choices=["light", 
                 "attention",
                 "full"],
    )
    parser.add_argument(
        "--quantized",
        help="Loading quantized model in 4-bit",
        action='store_true',
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        help="Path to a checkpoint directory to resume training from",
        default=None,
    )

    args = parser.parse_args()
    
    print("\n############### Hugging Face login: ###############\n")
    os.system('huggingface-cli whoami')
    with open("config.json") as f: config_data = json.load(f)
    HF_TOKEN = config_data["HF_TOKEN"]
    login(token = HF_TOKEN)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nDevice:", device)

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')

    model_id = args.model
    dataset_path = args.dataset_path
    quantize = args.quantized
    Lora_directory = args.lora_directory
    r = int(args.rank)
    alpha = int(args.lora_alpha)
    lr = float(args.learning_rate)
    scheduler = args.scheduler
    model_name = args.model_name
    lora_type = args.lora_type
    resume_from_checkpoint = args.resume_from_checkpoint

    if lora_type == 'light':
        target_modules = ["q_proj", "v_proj"]
    elif lora_type == 'attention':
        target_modules = ["q_proj", "v_proj", "o_proj", "k_proj"]
    elif lora_type == 'full':
        target_modules = ['down_proj', 'gate_proj', 'o_proj', 'v_proj', 'up_proj', 'q_proj', 'k_proj']
    dataset = load_from_disk(dataset_path)
    
    print('\n########## Directory used to save LoRA weights: ', Lora_directory)

    #################### TOKENIZER AND MODEL ####################
    tokenizer, model = load_tokenizer_and_model(quantize, model_id, HF_TOKEN)

    #################### PREPARING CONFIG FILES ####################
    peft_config, sft_config, wandb_config = set_config(target_modules, Lora_directory, r, alpha, lr, scheduler)

    model2 = get_peft_model(model, peft_config)
    model2.print_trainable_parameters()

    if device.type == 'cuda':
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')    
    train(tokenizer, model, sft_config, peft_config, wandb_config, dataset, model_name, resume_from_checkpoint)



if __name__ == "__main__":
    main()