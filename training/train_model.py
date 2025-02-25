import os
import json
import torch
import math
import hf_llm_args
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments
)
import text2query_llm_dataset
from transformers import set_seed
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

LLM_MAPPING_DICT = {
    "Qwen/Qwen2.5-Coder-0.5B-Instruct":{"response_template": "<|im_start|>assistant\n"},
    "Qwen/Qwen2.5-Coder-1.5B-Instruct":{"response_template": "<|im_start|>assistant\n"}
}

if __name__ == "__main__":
    parser = HfArgumentParser(hf_llm_args.ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    set_seed(args.seed)

    run_name = args.run_name
    path_to_training_file = args.path_to_training_file
    path_to_testing_file = args.path_to_testing_file
    model_path = args.model_name_or_path
    output_dir = args.output_dir
    overwrite_output_dir = args.overwrite_output_dir
    report_to = args.report_to
    logging_dir = args.logging_dir

    num_train_epochs = args.num_train_epochs
    try_one_batch = args.try_one_batch
    per_device_train_batch_size = args.per_device_train_batch_size
    per_device_eval_batch_size = args.per_device_eval_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    learning_rate = args.learning_rate
    max_seq_length = args.max_seq_length
    use_lora = args.use_lora
    eval_steps = args.eval_steps
    logging_steps = args.logging_steps



    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    new_rubq_tokens = ['wdt:', 'skos:', 'wd:', 'ps:', 'pq:']

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_tokens(new_rubq_tokens)

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map=device)
    model.resize_token_embeddings(len(tokenizer))

    print('Loaded model!')

    # read data
    training_sft_dataset = json.load(open(path_to_training_file, 'r'))
    # validation_sft_dataset = json.load(open(path_to_testing_file, 'r'))
    validation_sft_dataset = json.load(open(path_to_training_file, 'r'))

    if try_one_batch:
        training_sft_dataset = training_sft_dataset[:per_device_train_batch_size]
        validation_sft_dataset = training_sft_dataset[:per_device_train_batch_size]

    tokenized_train_sft_dataset = text2query_llm_dataset.LlmFinetuneDataset(sft_dataset=training_sft_dataset,
                                                                            device=device, tokenizer=tokenizer,
                                                                           max_sft_length=max_seq_length)
    tokenized_validation_sft_dataset = text2query_llm_dataset.LlmFinetuneDataset(sft_dataset=validation_sft_dataset,
                                                                           device=device, tokenizer=tokenizer,
                                                                           max_sft_length=max_seq_length)

    print('Training samples total size: ', len(tokenized_train_sft_dataset))

    peft_config = None
    if use_lora:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=128,
            bias="none",
            target_modules=['q_proj', 'v_proj',
                            'k_proj', 'o_proj',
                            'gate_proj',
                            'up_proj', 'down_proj'],
            task_type="CAUSAL_LM",
        )

    batch_size = per_device_train_batch_size * gradient_accumulation_steps
    num_update_steps_per_epoch = max(len(training_sft_dataset) // batch_size, 1)
    total_train_steps = math.ceil(num_train_epochs * num_update_steps_per_epoch)
    num_warmup_steps = int(0.03 * total_train_steps)
    print('My total train steps: ', total_train_steps)


    response_template = LLM_MAPPING_DICT[model_path]['response_template']
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    # The assistant answer is ignored during loss calculation
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch",
        save_steps=eval_steps,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        learning_rate=learning_rate,
        bf16=True,
        max_grad_norm=0.3,
        num_train_epochs=num_train_epochs,
        warmup_steps=num_warmup_steps,
        lr_scheduler_type="cosine",
        report_to=report_to,
        overwrite_output_dir=overwrite_output_dir,
        logging_dir=logging_dir,
        logging_strategy='steps',
        run_name=run_name,
        save_total_limit=1
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_train_sft_dataset,
        eval_dataset=tokenized_validation_sft_dataset,
        peft_config=peft_config,
        dataset_text_field="sft",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        data_collator=collator
    )
    print('Begin training!')
    trainer.train()
    print(f'Training finished, saving to {output_dir}')

    output_dir = os.path.join(output_dir, "final_checkpoints")
    trainer.model.save_pretrained(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)
