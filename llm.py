from helper import *
import sys
from datasets import Dataset
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from unsloth import FastModel
import torch
import numpy as np
import os
import sys
import re

# Disable tokenizers parallelism warning when using DataLoader with multiple workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_and_evaluate(
    train_data_raw,
    test_data_raw,
    subtask=3,
    domain="restaurant",
    model_name_or_path="unsloth/gemma-3-27b-it-bnb-4bit",
    num_train_epochs=5,
    train_batch_size=4,
    max_seq_length=512,
    learning_rate=2e-4,
    seed=42,
    lora_rank=64,
    lora_alpha=16,
):
    # Set random seed for reproducibility
    set_seed(seed)

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=max_seq_length,
        # load_in_4bit=True,
        # load_in_8bit=False,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=seed,
    )

    # Convert label objects to tuples for training
    from helper import get_prompt
    
    for example in train_data_raw:
        if "label" in example:
            # Convert label objects to tuples
            example["label"] = convert_label_objects_to_tuples(example["label"], subtask=subtask)
        
        prompt = get_prompt(
            text=example["text"],
            subtask=subtask,
            domain=domain
        )
        example["text"] = "<start_of_turn>user\n" + prompt + \
            "<end_of_turn>\n<start_of_turn>model\n" + \
            str(example["label"]) + "<end_of_turn>"
    print(example["text"])

    train_data = Dataset.from_list(train_data_raw)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=seed,
            report_to="none",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    trainer.train()

    # Save model and tokenizer to model_temp
    print("Saving model and tokenizer to model_temp...")
    model.save_pretrained("model_temp")
    tokenizer.save_pretrained("model_temp")

    # perform evaluation here using vLLM

    print("Initializing vLLM for evaluation...")
    # Initialize vLLM with the base model
    llm = LLM(
        model=model_name_or_path,
        tokenizer=model_name_or_path,
        enable_lora=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.7
    )

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=max_seq_length)

    print(len(test_data_raw), "examples to evaluate")

    # Prepare prompts for batch inference
    prompts = []
    for example in test_data_raw:
        prompt = get_prompt(
            text=example["text"],
            subtask=subtask,
            domain=domain
        )
        prompt = "<start_of_turn>user\n" + prompt + \
            "<end_of_turn>\n<start_of_turn>model\n"
        prompts.append(prompt)

    # Generate with the LoRA adapter
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest("adapter", 1, "model_temp")
    )

    all_preds_formatted = []

    for idx, output in enumerate(outputs):
        try:
            raw_output = output.outputs[0].text
            parsed_tuples = parse_label_string(raw_output, subtask=subtask)
            
            # Convert to output format for JSONL submission
            formatted_output = convert_tuples_to_output_format(
                parsed_tuples, 
                test_data_raw[idx]["id"], 
                subtask=subtask
            )
            all_preds_formatted.append(formatted_output)
        except Exception as e:
            print(f"Error parsing output {idx}: {e}")
            # Empty prediction in output format
            if subtask == 3:
                all_preds_formatted.append({"ID": test_data_raw[idx]["id"], "Quadruplet": []})
            else:
                all_preds_formatted.append({"ID": test_data_raw[idx]["id"], "Triplet": []})

    return all_preds_formatted
