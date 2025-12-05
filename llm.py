from helper import *
import sys
from datasets import Dataset
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import StructuredOutputsParams
from unsloth import FastModel
import torch
import numpy as np
import os
import sys
import re
import logging

logger = logging.getLogger('training')

# Disable tokenizers parallelism warning when using DataLoader with multiple workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_PRED_SC = 15


def train_and_evaluate(
    train_data_raw,
    test_data_raw,
    dev_data_raw=None,
    subtask=3,
    language="eng",
    domain="restaurant",
    model_name_or_path="unsloth/gemma-3-27b-it-bnb-4bit",
    num_train_epochs=5,
    train_batch_size=4,
    max_seq_length=1024,
    learning_rate=2e-4,
    seed_run=42,
    lora_rank=64,
    lora_alpha=16,
    strategy="evaluation",
    split_idx=0,
):
    # Set random seed for reproducibility
    set_seed(seed_run)

    logger.info(
        f"Starting training - Subtask: {subtask}, Language: {language}, Domain: {domain}, Seed: {seed_run}")
    logger.info(
        f"Training parameters - Epochs: {num_train_epochs}, Batch size: {train_batch_size}, LR: {learning_rate}, LoRA rank: {lora_rank}")
    logger.info(
        f"Training examples: {len(train_data_raw)}, Test examples: {len(test_data_raw) if test_data_raw is not None else 'N/A'}, Dev examples: {len(dev_data_raw ) if dev_data_raw is not None else 'N/A'}")

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
        random_state=seed_run,
    )

    # Determine chat template tags based on model
    if "gemma" in model_name_or_path.lower():
        user_start = "<start_of_turn>user\n"
        user_end = "<end_of_turn>\n"
        model_start = "<start_of_turn>model\n"
        model_end = "<end_of_turn>"
    else:  # Qwen and others
        user_start = "<|im_start|>user\n"
        user_end = "<|im_end|>\n"
        model_start = "<|im_start|>assistant\n"
        model_end = "<|im_end|>"

    for example in train_data_raw:
        if "label" in example:
            # Convert label objects to tuples
            example["label"] = convert_label_objects_to_tuples(
                example["label"], subtask=subtask)

        prompt = get_prompt(
            text=example["text"],
            subtask=subtask,
            language=language,
            domain=domain
        )
        example["text"] = user_start + prompt + \
            user_end + model_start + \
            str(example["label"]) + model_end

    # Log first example prompt for debugging
    logger.info("Example prompt (first training example):")
    logger.info(f"{example['text']}...")

    train_data = Dataset.from_list(train_data_raw)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3" if "gemma" in model_name_or_path else "qwen-3",
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
            seed=seed_run,
            report_to="none",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part=user_start,
        response_part=model_start,
    )

    logger.info("Starting model training...")
    start_time_training = datetime.now()
    trainer.train()
    total_time_training = datetime.now() - start_time_training
    logger.info(f"Total training time: {total_time_training}")
    logger.info("Training completed")
    
    # store time of training in time_logs.jsonl
    with open("time_logs.jsonl", "a") as f:
        log_entry = {
            "subtask": subtask,
            "language": language,
            "domain": domain,
            "seed_run": seed_run,
            "strategy": strategy,
            "split_idx": split_idx,
            "model_name_or_path": model_name_or_path,
            "num_train_epochs": num_train_epochs,
            "train_batch_size": train_batch_size,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "training_time": total_time_training.total_seconds(),
            "timestamp": datetime.now().isoformat()
        }
        f.write(json.dumps(log_entry) + "\n")

    # Save model and tokenizer to model_temp
    logger.info("Saving model and tokenizer to model_temp...")
    model.save_pretrained("model_temp")
    tokenizer.save_pretrained("model_temp")

    # perform evaluation here using vLLM
    
    setup_gpu_environment()
    clear_memory()

    logger.info("Initializing vLLM for evaluation...")
    # Initialize vLLM with the base model
    llm = LLM(
        model=model_name_or_path,
        tokenizer=model_name_or_path,
        enable_lora=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.7,
        seed=seed_run,
        max_num_seqs=512,
        max_model_len=max_seq_length,
        guided_decoding_backend="xgrammar",
    )

    # Create sampling parameters

    logger.info(f"Evaluating {len(test_data_raw) if test_data_raw is not None else 'N/A'} examples and evaluating {len(dev_data_raw) if dev_data_raw is not None else 'N/A'} dev examples...")

    # If strategy is "pred_dev" or "evaluation", evaluate on dev set
    if strategy == "train_split":
        evaluate_model(test_data_raw, subtask, language,
                   domain, llm, seed_run, strategy, model_name_or_path, split_idx=split_idx)
    
    if strategy == "evaluation":
        evaluate_model(dev_data_raw, subtask, language,
                       domain, llm, seed_run, "pred_dev", model_name_or_path, split_idx=split_idx)
    
    if test_data_raw is not None and strategy == "evaluation":
        evaluate_model(test_data_raw, subtask, language,
                       domain, llm, seed_run, "pred_test", model_name_or_path, split_idx=split_idx)
    
    

def evaluate_chunked(prompts, sampling_params, llm, lora_request=None, chunks=1500):
    if len(prompts) <= chunks:
        return llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            lora_request=lora_request
        )
    all_outputs = []
    for i in range(0, len(prompts), chunks):
        chunk_prompts = prompts[i:i+chunks]
        chunk_sampling_params = sampling_params[i:i+chunks]
        outputs = llm.generate(
            prompts=chunk_prompts,
            sampling_params=chunk_sampling_params,
            lora_request=LoRARequest("adapter", 1, "model_temp")
        )
        all_outputs.extend(outputs)
    return all_outputs


def store_evaluation_time(subtask, language, domain, seed_run, strategy, model_name_or_path, split_idx, evaluation_time, self_consistency=False, guided=False):
    with open("time_logs.jsonl", "a") as f:
        log_entry = {
            "subtask": subtask,
            "language": language,
            "domain": domain,
            "seed_run": seed_run,
            "strategy": strategy,
            "split_idx": split_idx,
            "model_name_or_path": model_name_or_path,
            "evaluation_time": evaluation_time.total_seconds(),
            "timestamp": datetime.now().isoformat(),
            "self_consistency": self_consistency,
            "guided": guided
        }
        f.write(json.dumps(log_entry) + "\n")

def evaluate_model(evaluation_set_raw, subtask, language, domain, llm, seed_run, strategy, model_name_or_path, split_idx=0):
    logger.info(f"Starting evaluation - Strategy: {strategy}, Subtask: {subtask}, Language: {language}, Domain: {domain}")
    logger.info(f"Evaluation set size: {len(evaluation_set_raw)} examples")

    # Determine chat template tags based on model
    if "gemma" in model_name_or_path.lower():
        user_start = "<start_of_turn>user\n"
        user_end = "<end_of_turn>\n"
        model_start = "<start_of_turn>model\n"
    else:  # Qwen and others
        user_start = "<|im_start|>user\n"
        user_end = "<|im_end|>\n"
        model_start = "<|im_start|>assistant\n"

    prompts = []
    for example in evaluation_set_raw:
        prompt = get_prompt(
            text=example["text"],
            subtask=subtask,
            language=language,
            domain=domain
        )
        prompt = user_start + prompt + \
            user_end + model_start
        prompts.append(prompt)
        
    unique_aspect_categories = get_unique_aspect_categories(domain)
    polarities = ["positive", "negative", "neutral"]

    # Determine disable_null_aspect: False for jpn+hotel, or if strategy is train_split
    disable_null_aspect = True
    if (language == "jpn" and domain == "hotel") or strategy == "train_split":
        disable_null_aspect = False

    # 1a. Prediction mit temp=0 ohne guided decoding
    # 1b. Prediction mit temp=0 mit guided decoding
    logger.info("Preparing sampling parameters with guided decoding patterns...")

    sampling_params_list = []
    for i, _ in enumerate(prompts):
        # unique_aspect_categories is the list of all aspect categories in the dataset
        pattern = get_regex_pattern_tuple(
            unique_aspect_categories, polarities, evaluation_set_raw[i]["text"], subtask=subtask, disable_null_aspect=disable_null_aspect)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            structured_outputs=StructuredOutputsParams(regex=pattern),
            seed=seed_run
        )
        sampling_params_list.append(sampling_params)

    # Generate with the LoRA adapter
    logger.info("Running inference 1a: temp=0, no guidance...")
    start_time_1a = datetime.now()
    outputs_1a = llm.generate(
        prompts=prompts,
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=512,
            seed=seed_run
        ),
        lora_request=LoRARequest("adapter", 1, "model_temp")
    )
    total_time_1a = datetime.now() - start_time_1a
    store_evaluation_time(subtask, language, domain, seed_run, strategy, model_name_or_path, split_idx, total_time_1a, self_consistency=False, guided=False)
    logger.info(f"Inference 1a completed in {total_time_1a}")

    logger.info("Running inference 1b: temp=0, with guidance...")
    start_time_1b = datetime.now()
    outputs_1b = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params_list,
        lora_request=LoRARequest("adapter", 1, "model_temp"),
    )
    total_time_1b = datetime.now() - start_time_1b
    store_evaluation_time(subtask, language, domain, seed_run, strategy, model_name_or_path, split_idx, total_time_1b, self_consistency=False, guided=True)
    logger.info(f"Inference 1b completed in {total_time_1b}")

    outputs_1a = format_predictions(outputs_1a, subtask, evaluation_set_raw, disable_null_aspect=disable_null_aspect)
    outputs_1b = format_predictions(outputs_1b, subtask, evaluation_set_raw, disable_null_aspect=disable_null_aspect)

    # 2a. Prediction mit temp=0.8 -> NUM_PRED_SC mal gleiche prompt ausführen ohne guided decoding
    logger.info("Running inference 2a: temp=0.8, NUM_PRED_SC runs, no guidance...")
    prompts_2a = prompts * NUM_PRED_SC
    sampling_params_2a = []
    for k in range(NUM_PRED_SC):
        for _ in prompts:
            sampling_params_2a.append(SamplingParams(
                temperature=0.8,
                max_tokens=512,
                seed=k
            ))
    
    start_time_2a = datetime.now()
    outputs_2a = llm.generate(
        prompts=prompts_2a,
        sampling_params=sampling_params_2a,
        lora_request=LoRARequest("adapter", 1, "model_temp")
    )
    total_time_2a = datetime.now() - start_time_2a
    store_evaluation_time(subtask, language, domain, seed_run, strategy, model_name_or_path, split_idx, total_time_2a, self_consistency=True, guided=False)
    logger.info(f"Inference 2a completed in {total_time_2a}")
        
    # 2b. Prediction mit temp=0.8 -> NUM_PRED_SC mal gleiche prompt ausführen mit guided decoding
    # Pattern einmal pro Prompt berechnen und für alle NUM_PRED_SC Runs wiederverwenden
    logger.info("Running inference 2b: temp=0.8, NUM_PRED_SC runs, with guidance (chunked)...")
    patterns = []
    for i, _ in enumerate(prompts):
        pattern = get_regex_pattern_tuple(
            unique_aspect_categories, polarities, evaluation_set_raw[i]["text"], subtask=subtask, disable_null_aspect=disable_null_aspect)
        patterns.append(pattern)
    
    prompts_2b = prompts * NUM_PRED_SC
    sampling_params_2b = []
    for k in range(NUM_PRED_SC):
        for i, _ in enumerate(prompts):
            sampling_params_2b.append(SamplingParams(
                temperature=0.8,
                max_tokens=512,
                structured_outputs=StructuredOutputsParams(regex=patterns[i]),
                seed=k
            ))
    
    start_time_2b = datetime.now()
    outputs_2b = evaluate_chunked(prompts_2b, sampling_params_2b, llm, lora_request=LoRARequest("adapter", 1, "model_temp"), chunks=1500)
    total_time_2b = datetime.now() - start_time_2b
    store_evaluation_time(subtask, language, domain, seed_run, strategy, model_name_or_path, split_idx, total_time_2b, self_consistency=True, guided=True)
    logger.info(f"Inference 2b completed in {total_time_2b}")

    outputs_2a = format_predictions(outputs_2a, subtask, evaluation_set_raw * NUM_PRED_SC, disable_null_aspect=disable_null_aspect)
    outputs_2b = format_predictions(outputs_2b, subtask, evaluation_set_raw * NUM_PRED_SC, disable_null_aspect=disable_null_aspect)

    # create results directory if not exists
    if not os.path.exists(f"results/results_{strategy}/{model_name_or_path.replace('/', '_')}"):
        os.makedirs(
            f"results/results_{strategy}/{model_name_or_path.replace('/', '_')}")

    if strategy == "train_split":
        path_1a = f"results/results_{strategy}/{model_name_or_path.replace('/', '_')}/{subtask}_{language}_{domain}_{seed_run}_{split_idx}_temp0_no_guidance.jsonl"
        with open(path_1a, "w") as f:
            for item in outputs_1a:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        path_1b = f"results/results_{strategy}/{model_name_or_path.replace('/', '_')}/{subtask}_{language}_{domain}_{seed_run}_{split_idx}_temp0_with_guidance.jsonl"
        with open(path_1b, "w") as f:
            for item in outputs_1b:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        for i in range(NUM_PRED_SC):
            path_2a = f"results/results_{strategy}/{model_name_or_path.replace('/', '_')}/{subtask}_{language}_{domain}_{seed_run}_{split_idx}_temp0.8_no_guidance_run{i}.jsonl"
            with open(path_2a, "w") as f:
                for item in outputs_2a[i*len(prompts):(i+1)*len(prompts)]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            path_2b = f"results/results_{strategy}/{model_name_or_path.replace('/', '_')}/{subtask}_{language}_{domain}_{seed_run}_{split_idx}_temp0.8_with_guidance_run{i}.jsonl"
            with open(path_2b, "w") as f:
                for item in outputs_2b[i*len(prompts):(i+1)*len(prompts)]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
    elif strategy == "pred_dev" or strategy == "pred_test":
        path_1a = f"results/results_{strategy}/{model_name_or_path.replace('/', '_')}/{subtask}_{language}_{domain}_{seed_run}_temp0_no_guidance.jsonl"
        with open(path_1a, "w") as f:
            for item in outputs_1a:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        path_1b = f"results/results_{strategy}/{model_name_or_path.replace('/', '_')}/{subtask}_{language}_{domain}_{seed_run}_temp0_with_guidance.jsonl"
        with open(path_1b, "w") as f:
            for item in outputs_1b:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        for i in range(NUM_PRED_SC):
            path_2a = f"results/results_{strategy}/{model_name_or_path.replace('/', '_')}/{subtask}_{language}_{domain}_{seed_run}_temp0.8_no_guidance_run{i}.jsonl"
            with open(path_2a, "w") as f:
                for item in outputs_2a[i*len(prompts):(i+1)*len(prompts)]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            path_2b = f"results/results_{strategy}/{model_name_or_path.replace('/', '_')}/{subtask}_{language}_{domain}_{seed_run}_temp0.8_with_guidance_run{i}.jsonl"
            with open(path_2b, "w") as f:
                for item in outputs_2b[i*len(prompts):(i+1)*len(prompts)]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")


def format_predictions(outputs, subtask, test_data_raw, disable_null_aspect=True):
    all_preds_formatted = []

    for idx, output in enumerate(outputs):
        try:
            raw_output = output.outputs[0].text
            parsed_tuples = parse_label_string(raw_output, subtask=subtask)
            if len(parsed_tuples) == 0:
                logger.warning(
                    f"No tuples parsed from output {idx}: {raw_output}")

            # Convert to output format for JSONL submission
            formatted_output = convert_tuples_to_output_format(
                parsed_tuples,
                test_data_raw[idx]["id"],
                subtask=subtask,
                disable_null_aspect=disable_null_aspect
            )
            all_preds_formatted.append(formatted_output)
        except Exception as e:
            logger.warning(f"Error parsing output {idx}: {e}")
            # Empty prediction in output format
            if subtask == 3:
                all_preds_formatted.append(
                    {"ID": test_data_raw[idx]["id"], "Quadruplet": []})
            else:
                all_preds_formatted.append(
                    {"ID": test_data_raw[idx]["id"], "Triplet": []})

    return all_preds_formatted
