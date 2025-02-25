import os
import json
import torch
import pickle
import hf_llm_args
from tqdm import tqdm
import training_utils
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    AutoTokenizer,
)
import text2query_llm_dataset
from transformers import set_seed
from torch.utils.data import DataLoader
from peft import AutoPeftModelForCausalLM

if __name__ == "__main__":
    parser = HfArgumentParser(hf_llm_args.ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    set_seed(args.seed)

    sparql_dataset_name = args.sparql_dataset_name
    path_to_testing_file = args.path_to_testing_file
    model_name_or_path = args.model_name_or_path
    output_dir = args.output_dir

    try_one_batch = args.try_one_batch
    per_device_eval_batch_size = args.per_device_eval_batch_size
    max_seq_length = args.max_seq_length
    max_new_tokens = args.max_new_tokens
    num_beams = args.num_beams
    use_lora = args.use_lora


    device = 'cuda'
    # if torch.cuda.is_available():
    #     device = 'cuda'

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"
    terminators = [
        tokenizer.eos_token_id,
    ]

    if use_lora:
        model = AutoPeftModelForCausalLM.from_pretrained(model_name_or_path,
                                                         torch_dtype=torch.float32,
                                                         device_map=device)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                     torch_dtype=torch.float32,
                                                     device_map=device)

    model.generation_config.pad_token_ids = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))


    testing_sft_dataset = json.load(open(path_to_testing_file, 'r'))
    if try_one_batch:
        testing_sft_dataset = testing_sft_dataset[:per_device_eval_batch_size]

    tokenized_test_sft_dataset = text2query_llm_dataset.LlmFinetuneDataset(sft_dataset=testing_sft_dataset,
                                                                           device=device, tokenizer=tokenizer,
                                                                           max_sft_length=max_seq_length)

    print(f'Total testing samples = {len(tokenized_test_sft_dataset)}')

    test_dataloader = DataLoader(tokenized_test_sft_dataset, shuffle=False, batch_size=per_device_eval_batch_size)

    ids_list, prediction_list, scores_list = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            sample_id = batch['id']
            input_length = batch['input_ids'].shape[1]
            outputs = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                eos_token_id=terminators,
                output_logits=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )


            generated_sequences = outputs["sequences"].cpu() if "cuda" in device else outputs["sequences"]

            entropy_scores = training_utils.maximum_entropy_confidence_score_method(generation_scores=outputs["logits"],
                                                                                    device=device)
            entropy_scores = training_utils.truncate_scores(generated_sequences=generated_sequences,
                                                            scores=entropy_scores,
                                                            tokenizer=tokenizer)
            max_entropy_scores = [max(score_list) for score_list in entropy_scores]
            scores_list += max_entropy_scores
            decoded_preds = tokenizer.batch_decode(generated_sequences[:, input_length:],
                                                   skip_special_tokens=True, clean_up_tokenization_spaces=False)
            predictions = [pred for pred in decoded_preds]
            prediction_list += predictions
            ids_list += sample_id

    print('Inference completed!')

    result_dict = dict()
    for id_, pred_query, score in zip(ids_list, prediction_list, scores_list):
        result_dict[id_] = {
            "query": pred_query,
            "score": score
        }

    output_dir = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.basename(path_to_testing_file).split('.')[0]
    if try_one_batch:
        filename = f"{sparql_dataset_name}_one_batch_inference_result.pkl"
    else:
        filename = f"{sparql_dataset_name}_inference_result.pkl"
    save_path = os.path.join(output_dir, filename)

    pickle.dump(result_dict, open(save_path, 'wb'))

    filename = filename.split('.')[0]

    if try_one_batch:
        filename = f"{filename}_one_batch_query_predictions.txt"
    else:
        filename = f"{filename}_query_predictions.txt"
    save_path = os.path.join(output_dir, filename)
    with open(save_path, 'w') as f:
        for query in prediction_list:
            f.write(f"{query}\n")




