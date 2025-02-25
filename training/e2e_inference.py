import os
import json
from tqdm import tqdm
from transformers import (
    AutoTokenizer
)
from preprocessing_utils import preprocess_sparql
from SPARQLWrapper import SPARQLWrapper, JSON
import re
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from SPARQLWrapper import SPARQLWrapper, JSON, POST
import time


EN_INSTRUCTION = """You are an expert SPARQL query generator for Wikidata. Your task is to transform natural language questions into correct and efficient SPARQL queries, ensuring precise alignment with the question, the provided entities, relations, and valid triplets.You are given valid Wikidata triplets constructed from the provided entities and relations. These triplets follow the correct syntax and use the proper prefixes. Provided vlaid triplets are based solely on the given information; if the query requires connections or entities not explicitly provided, generate new triplets accordingly while maintaining valid Wikidata syntax.
Ensure proper structure and syntax of SPARQL query. Optimize queries for performance, applying filters, counts, and conditions when necessary. Output only the complete SPARQL query with correct formatting, without explanations or extra text. Handle missing entities, ambiguous cases, and complex queries logically.
Ensure the syntax correctness of generated SPARQL query. Double-check that queries have correctly closed parentheses and braces.

Use common namespaces and prefixes for Wikidata:
wd (Wikidata Item) - Refers to a specific Wikidata item by its unique QID (e.g., wd:Q42 for Douglas Adams).
wdt (Wikidata Truthy Property) - Refers to the direct, unqualified value of a property for an item (e.g., wd:Q42 wdt:P31 wd:Q5 to indicate Douglas Adams is a human).
p (Property) - Refers to a property in Wikidata, including its full statement (e.g., wd:P31 for "instance of").
ps (Property Statement) - Refers to the value of a specific property within a statement (e.g., ps:P69 to refer to the value of "educated at").
pq (Qualifier) - Refers to qualifiers providing additional details for a property statement (e.g., pq:P580 for "start time" in a statement).
pr (Reference) - Refers to references providing sourcing information for a statement (e.g., pr:P854 for a source URL).
"""

INSTRUCTIONS = {
    'en': EN_INSTRUCTION,
}

def get_entities(question):
    return {'Q118': {'en': 'Mestro'}}


def create_prompt(question, entities_string, predicates_string):
    #  return f'''Question: {question}\n\nEntities:\n{entities_string}\n\nRelations:\n{predicates_string}\n\nWikidata valid triplets:\n{valid_triplets}\n'''
    return f'''Question: {question}\n\nEntities:\n{entities_string}\n\nRelations:\n{predicates_string}\n'''

wikidata_relations_info = json.load(open('../data/wikidata/wikidata_relations_info.json'))

def format_predicates(predicates):
    predicates_list = []
    for wikidata_id, label in predicates.items():
        format =  f"[{label}] - ({wikidata_id})"
        predicates_list.append(format)
    return '\n'.join(predicates_list)

def format_entities(entities, lang='en'):
    entities_list = []
    for wikidata_id, label in entities.items():
        format =  f"[{label}] - ({wikidata_id})"
        entities_list.append(format)
    return '\n'.join(entities_list)

if __name__ == '__main__':
    dataset_name = 'rubq'

    predicates_data = json.load(open('e2e_dataset/rubq_result_properties_10.json'))
    entities_data = json.load(open('e2e_dataset/rubq_result_entities_10.json'))

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    sft_examples_list, failed_samples = [], []
    lang = 'en'
    mode = 'test'

    instruction = INSTRUCTIONS[lang]

    for id_, item in entities_data.items():
        question = item['question_eng']
        query = item['query']
        entities = item['entities']
        if id_ in predicates_data:
            predicates = predicates_data[id_]['candidates']
        else:
            continue

        entities_string = format_entities(entities)
        predicates_string = format_predicates(predicates)

        user_task = create_prompt(question, predicates_string, entities_string)

        # This SPARQL line is redundant
        sparql = query.replace(
            'SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }', '')
        sparql = preprocess_sparql(sparql)

        if mode == 'train':
            chat = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_task},
                {"role": "assistant", "content": sparql}
            ]
            formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        else:
            chat = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_task}
            ]
            formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        sft_examples_list.append({"id": str(id_), "sft": formatted_prompt})

        json.dump(sft_examples_list,
                  open(f"e2e_dataset/{dataset_name}_test.json", 'w'),
                  ensure_ascii=False, indent=4
                  )


