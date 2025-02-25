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


def build_union_query(entities, relations, prefixes):
    """
    Build a single SPARQL query that tests all candidate triplets.

    For each candidate combination (entity, relation, prefix, position),
    create a subquery branch that returns a candidate string if a matching triple exists.

    Instead of using a separate BIND, we inline the candidate string into the SELECT clause.
    """
    sparql_prefixes = (
        "PREFIX wd: <http://www.wikidata.org/entity/>\n"
        "PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
        "PREFIX p: <http://www.wikidata.org/prop/>\n"
        "PREFIX ps: <http://www.wikidata.org/prop/statement/>\n"
        "PREFIX pq: <http://www.wikidata.org/prop/qualifier/>\n"
    )

    union_branches = []

    for entity in entities:
        for relation in relations:
            for prefix in prefixes:
                # Branch: entity as subject.
                branch_subject = f"""
                {{
                  SELECT ( "wd:{entity} {prefix}:{relation} ?object" AS ?candidate ) WHERE {{
                    wd:{entity} {prefix}:{relation} ?dummy .
                  }} LIMIT 1
                }}
                """
                union_branches.append(branch_subject)

                # Branch: entity as object.
                branch_object = f"""
                {{
                  SELECT ( "?subject {prefix}:{relation} wd:{entity}" AS ?candidate ) WHERE {{
                    ?dummy {prefix}:{relation} wd:{entity} .
                  }} LIMIT 1
                }}
                """
                union_branches.append(branch_object)

    union_part = "\nUNION\n".join(union_branches)

    query = f"""
    {sparql_prefixes}
    SELECT DISTINCT ?candidate WHERE {{
      {union_part}
    }}
    """
    return query


def execute_query_with_retry(sparql, query, max_retries=3, initial_delay=3):
    """
    Executes a SPARQL query with a retry mechanism in case of HTTP 429 errors.
    Uses POST method to avoid issues with long URIs.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        sparql.setQuery(query)
        sparql.setMethod(POST)  # Use POST to send the query body
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            return results
        except Exception as e:
            if "429" in str(e):
                print(
                    f"HTTP Error 429 encountered. Sleeping for {delay} seconds before retrying (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff.
            else:
                continue
    print("Max retries exceeded for query:")
    print(query)
    return None


def get_valid_triplets_single_query(entities, relations):
    """
    Build and execute a single SPARQL query that checks which candidate triplets are valid.

    This version uses inline candidate expressions in the SELECT clause (instead of BIND)
    so that the query is accepted by the endpoint.
    """
    endpoint_url = "https://query.wikidata.org/sparql"
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setTimeout(20)  # Increase timeout if necessary.

    predicate_prefixes = ["wdt", "p", "ps", "pq"]

    query = build_union_query(entities, relations, predicate_prefixes)

    results = execute_query_with_retry(sparql, query)
    if results is None:
        return ""

    # Only include bindings that actually have a "candidate" key.
    valid_triplets = [
        binding["candidate"]["value"]
        for binding in results["results"]["bindings"]
        if "candidate" in binding
    ]
    return "\n".join(valid_triplets)



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


def create_prompt(question, entities_string, predicates_string, valid_triplets):
    #  return f'''Question: {question}\n\nEntities:\n{entities_string}\n\nRelations:\n{predicates_string}\n\nWikidata valid triplets:\n{valid_triplets}\n'''
    return f'''Question: {question}\n\nEntities:\n{entities_string}\n\nRelations:\n{predicates_string}\n'''

wikidata_relations_info = json.load(open('../data/wikidata/wikidata_relations_info.json'))

def format_predicates(predicates, lang='en'):
    predicates_list = []
    for wikidata_id, lang_dict in predicates.items():
        if lang_dict:
            label = lang_dict.get(lang)
            format =  f"[{label}] - ({wikidata_id})"
            predicates_list.append(format)
        else:
            predicates_list.append(f'- ({wikidata_id})')

    return '\n'.join(predicates_list)

def format_entities(entities, lang='en'):
    entities_list = []
    for wikidata_id, lang_dict in entities.items():
        if lang_dict:
            label = lang_dict.get(lang)
            format =  f"[{label}] - ({wikidata_id})"
            entities_list.append(format)
        else:
            entities_list.append(f'- ({wikidata_id})')
    return '\n'.join(entities_list)


def format_dataset(dataset, tokenizer, mode='train', lang='en'):
    sft_examples_list, failed_samples = [], []

    instruction = INSTRUCTIONS[lang]
    for sample in tqdm(dataset):
        question = sample[f'{lang}_question']

        entities = {}
        if mode == 'train':
            entities = sample['entities'].get('question') or sample['entities']['query']
        elif mode == 'test':
            #entities = get_entities(question)
            entities = sample['entities'].get('question') or sample['entities']['query']

        predicates = {}
        if mode == 'train':
            predicates = sample['relations'].get('question') or sample['relations']['query']
        elif mode == 'test':
            predicates = sample['relations'].get('question') or sample['relations']['query']


        entities_string = format_entities(entities)
        predicates_string = format_predicates(predicates)

        # valid_triplets = get_valid_triplets_single_query(list(entities.keys()), list(predicates.keys()))
        valid_triplets = None

        user_task = create_prompt(question, entities_string, predicates_string, valid_triplets)

        # This SPARQL line is redundant
        sparql = sample['query'].replace('SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }', '')
        sparql = preprocess_sparql(sparql)

        if not entities_string or not sparql:
            failed_samples.append(sample)
            continue

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

        sft_examples_list.append({"id": str(sample['id']), "sft": formatted_prompt})

    return sft_examples_list, failed_samples

if __name__ == '__main__':
    dataset_name = 'rubq'

    train_data = json.load(open(f"../data/preprocessed/{dataset_name}/{dataset_name}_train.json"))
    test_data = json.load(open(f"../data/preprocessed/{dataset_name}/{dataset_name}_test.json"))

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    train_sft_examples_list, train_failed_samples = format_dataset(train_data['dataset'], tokenizer, mode='train')
    test_sft_examples_list, test_failed_samples = format_dataset(test_data['dataset'], tokenizer, mode='test')

    json.dump(train_sft_examples_list,
              open(f"sft/{dataset_name}_train.json", 'w'),
              ensure_ascii=False, indent=4
              )
    json.dump(test_sft_examples_list,
              open(f"sft/{dataset_name}_test.json", 'w'),
              ensure_ascii=False, indent=4
              )

    print('Prepared SFT train samples: ', len(train_sft_examples_list))
    print('Total of train failed samples: ', len(train_failed_samples))

    print('Prepared SFT train samples: ', len(test_sft_examples_list))
    print('Total of test failed samples: ', len(test_failed_samples))

