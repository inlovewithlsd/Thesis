import re
import json
import asyncio
import aiohttp
from tqdm import tqdm
from aiohttp import ClientSession
from sklearn.model_selection import train_test_split

from preprocessing_utils import (
    extract_wikidata_id_from_link,
    fetch_wikidata_labels,
    clean_sparql,
    map_wikidata_urls_to_prefix,
    fetch_with_semaphore,
    execute_wikidata_sparql_query
)

DATASETS_INFO = {
    'rubq': {
        'train': '../data/raw/rubq/rubq_train.json',
        'test': '../data/raw/rubq/rubq_test.json'
    },
    'qald': {
        'train': '../data/raw/qald/qald_train.json',
        'test': '../data/raw/qald/qald_test.json'
    },
    'lcquad_2.0': {
        'train': '../data/raw/lcquad/lcquad_2_train.json',
        'test': '../data/raw/lcquad/lcquad_2_test.json'
    },
    'pat': {
        'train': '../data/raw/pat/custom_iid_pat_train.json',
        'test': '../data/raw/pat/custom_iid_pat_test.json'
    },
}

def load_data(file_path):
    """Load data from the given file path."""
    with open(file_path, 'r') as f:
        return json.load(f)

async def fetch_and_cache_labels(session, entities, cached_labels):
    """Fetch labels for a set of entities and update the cache."""
    new_labels = entities - cached_labels.keys()
    if new_labels:
        tasks = [fetch_with_semaphore(session, entity) for entity in new_labels]
        results = await asyncio.gather(*tasks)
        cached_labels.update({entity: result for entity, result in zip(new_labels, results) if result})

async def preprocess_rubq(split):
    dataset_name = 'rubq'
    file_path = DATASETS_INFO[dataset_name][split]

    # Load data
    with open(file_path, 'r') as f:
        data = json.load(f)

    dataset = []
    all_entities = set()
    all_relations = set()
    cached_labels = {}

    try:
        async with ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            for item in tqdm(data):
                query = item.get('query')
                if not query:  # Skip if query is None
                    continue

                # Extract and deduplicate entities
                question_entities = list(
                    set(extract_wikidata_id_from_link(uri) for uri in item.get('question_uris', [])))
                query_entities = list(set(re.findall(r'(Q\d+)', query)))
                entities_to_fetch = set(question_entities + query_entities)
                all_entities.update(entities_to_fetch)

                # Extract and deduplicate relations
                question_relations = list(set(
                    match.group(1) for value in item.get('question_props', [])
                    if (match := re.search(r'(P\d+)', value))
                ))
                query_relations = list(set(re.findall(r'(P\d+)', query)))
                relations_to_fetch = set(question_relations + query_relations)
                all_relations.update(relations_to_fetch)

                # Fetch and cache labels for entities and relations
                await fetch_and_cache_labels(session, entities_to_fetch.union(relations_to_fetch), cached_labels)

                # Extract answers
                answer_entities = [extract_wikidata_id_from_link(ans['value']) for ans in item.get('answers', [])]

                # Fetch and cache labels for answer entities
                await fetch_and_cache_labels(session, set(answer_entities), cached_labels)

                # Prepare sample
                sample = {
                    'id': item['uid'],
                    'en_question': item.get('question_eng'),
                    'ru_question': item.get('question_text'),
                    'query': clean_sparql(query),
                    'entities': {
                        'question': {entity: cached_labels.get(entity) for entity in question_entities},
                        'query': {entity: cached_labels.get(entity) for entity in query_entities}
                    },
                    'relations': {
                        'question': {relation: cached_labels.get(relation) for relation in question_relations},
                        'query': {relation: cached_labels.get(relation) for relation in query_relations}
                    },
                    'answer_en': [cached_labels[entity].get('en') for entity in answer_entities if
                                  cached_labels.get(entity, {}).get('en')],
                    'answer_ru': [cached_labels[entity].get('ru') for entity in answer_entities if
                                  cached_labels.get(entity, {}).get('ru')],
                    'answer_entities': answer_entities
                }
                dataset.append(sample)

    except Exception as e:
        print(f"Error processing item: {item}")
        raise e

    finally:
        # Ensure data is saved even if an error occurs
        output_path = f'../data/preprocessed/{dataset_name}/{dataset_name}_{split}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': dataset,
                'entities': list(all_entities),
                'relations': list(all_relations)
            }, f, ensure_ascii=False, indent=4)
        print(f"Data saved to {output_path}")

async def preprocess_qald(split):
    dataset_name = 'qald'
    file_path = DATASETS_INFO[dataset_name][split]

    with open(file_path, 'r') as f:
        data = json.load(f)['questions']

    dataset = []
    all_entities = set()
    all_relations = set()
    cached_labels = {}

    try:
        async with ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            for item in tqdm(data):
                query = item.get('query').get('sparql')
                accepted_languages = [q['language'] for q in item['question']]
                if not query or not ('en' in accepted_languages and 'ru' in accepted_languages):
                    continue

                # clean query from prefixes and urls
                query = map_wikidata_urls_to_prefix(query)

                # Extract and deduplicate entities
                query_entities = list(set(re.findall(r'(Q\d+)', query)))
                entities_to_fetch = set(query_entities)
                all_entities.update(entities_to_fetch)

                # Extract and deduplicate relations
                query_relations = list(set(re.findall(r'(P\d+)', query)))
                relations_to_fetch = set(query_relations)
                all_relations.update(relations_to_fetch)

                # Fetch and cache labels for entities and relations
                await fetch_and_cache_labels(session, entities_to_fetch.union(relations_to_fetch), cached_labels)

                # Extract answers
                # answer_entities = list(*(extract_answers_from_response(response) for response in item['answers']))
                answer_entities = execute_wikidata_sparql_query(query)
                if not answer_entities:
                    continue

                # Fetch and cache labels for answer entities
                await fetch_and_cache_labels(session, set(answer_entities), cached_labels)

                # Prepare sample
                sample = {
                    'id': item['id'],
                    'en_question': next(filter(lambda q: q['language'] == 'en', item['question']))['string'],
                    'ru_question': next(filter(lambda q: q['language'] == 'ru', item['question']))['string'],
                    'query': clean_sparql(query),
                    'entities': {
                        'question': None,
                        'query': {entity: cached_labels.get(entity) for entity in query_entities}
                    },
                    'relations': {
                        'question': None,
                        'query': {relation: cached_labels.get(relation) for relation in query_relations}
                    },
                    'answer_en': [cached_labels[entity].get('en') for entity in answer_entities if
                                  cached_labels.get(entity, {}).get('en')],
                    'answer_ru': [cached_labels[entity].get('ru') for entity in answer_entities if
                                  cached_labels.get(entity, {}).get('ru')],
                    'answer_entities': answer_entities
                }
                dataset.append(sample)

    except Exception as e:
        print(f"Error processing item: {item}")
        raise e

    finally:
        # Ensure data is saved even if an error occurs
        output_path = f'../data/preprocessed/{dataset_name}/{dataset_name}_{split}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': dataset,
                'entities': list(all_entities),
                'relations': list(all_relations)
            }, f, ensure_ascii=False, indent=4)
        print(f"Data saved to {output_path}")



async def preprocess_lcquad(split):
    dataset_name = 'lcquad_2.0'
    file_path = DATASETS_INFO[dataset_name][split]

    # Load data
    with open(file_path, 'r') as f:
        data = json.load(f)

    dataset = []
    all_entities = set()
    all_relations = set()
    cached_labels = {}

    try:
        async with ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            for i, item in enumerate(tqdm(data)):
                query = item.get('query')
                if not query:
                    continue

                # Extract and deduplicate entities
                query_entities = list(set(re.findall(r'(Q\d+)', query)))
                entities_to_fetch = set(query_entities)
                all_entities.update(entities_to_fetch)

                # Extract and deduplicate relations
                query_relations = list(set(re.findall(r'(P\d+)', query)))
                relations_to_fetch = set(query_relations)
                all_relations.update(relations_to_fetch)

                # Fetch and cache labels for entities and relations
                await fetch_and_cache_labels(session, entities_to_fetch.union(relations_to_fetch), cached_labels)

                # Extract answers
                answer_entities = execute_wikidata_sparql_query(query)
                if not answer_entities:
                    continue

                # Fetch and cache labels for answer entities
                await fetch_and_cache_labels(session, set(answer_entities), cached_labels)

                # Prepare sample
                sample = {
                    'id': i,
                    'en_question': item.get('en_question'),
                    'ru_question': item.get('ru_question'),
                    'query': clean_sparql(query),
                    'entities': {
                        'question': None,
                        'query': {entity: cached_labels.get(entity) for entity in query_entities}
                    },
                    'relations': {
                        'question': None,
                        'query': {relation: cached_labels.get(relation) for relation in query_relations}
                    },
                    'answer_en': [cached_labels.get(entity).get('en') for entity in answer_entities if cached_labels.get(entity, {}).get('en')],
                    'answer_ru': [cached_labels.get(entity).get('ru') for entity in answer_entities if cached_labels.get(entity, {}).get('ru')],
                    'answer_entities': answer_entities
                }
                dataset.append(sample)

    except Exception as e:
        print(f"Error processing item: {item}")
        raise e

    finally:
        # Ensure data is saved even if an error occurs
        output_path = f'../data/preprocessed/{dataset_name}/{dataset_name}_{split}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': dataset,
                'entities': list(all_entities),
                'relations': list(all_relations)
            }, f, ensure_ascii=False, indent=4)
        print(f"Data saved to {output_path}")

async def preprocess_pat():
    dataset = []
    all_entities = set()
    all_relations = set()
    cached_labels = {}

    pat_singlehop = json.load(open('data/raw/pat/PAT-singlehop.json'))
    pat_multihop = json.load(open('data/raw/pat/PAT-multihop.json'))

    pat_data = pat_singlehop.copy()
    pat_data.update(pat_multihop)

    try:
        async with ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            for question, item in tqdm(pat_data.items()):
                query = item.get('query')
                if not query:
                    continue

                question_entities = [item['subject']['subject']]
                query_entities = list(set(re.findall(r'(Q\d+)', query)))
                entities_to_fetch = set(question_entities + query_entities)
                all_entities.update(entities_to_fetch)

                # Extract and deduplicate relations
                question_relations = item['relations']
                query_relations = list(set(re.findall(r'(P\d+)', query)))
                relations_to_fetch = set(question_relations + query_relations)
                all_relations.update(relations_to_fetch)

                await fetch_and_cache_labels(session, entities_to_fetch.union(relations_to_fetch), cached_labels)

                # Extract answers
                answer_entities = [ans.get('ID') for ans in item.get('answer annotations', []) if 'ID' in ans]
                answer_en = [ans.get('Label') for ans in item.get('answer annotations', []) if 'Label' in ans]

                # Fetch and cache labels for answer entities
                await fetch_and_cache_labels(session, set(answer_entities), cached_labels)

                # Prepare sample
                sample = {
                    'id': item['uniq_id'],
                    'en_question': question,
                    'ru_question': None,
                    'query': query,
                    'entities': {
                        'question': {entity: cached_labels.get(entity) for entity in question_entities},
                        'query': {entity: cached_labels.get(entity) for entity in query_entities}
                    },
                    'relations': {
                        'question': {relation: cached_labels.get(relation) for relation in question_relations},
                        'query': {relation: cached_labels.get(relation) for relation in query_relations}
                    },
                    'answer_en': answer_en,
                    'answer_ru': None,
                    'answer_entities': answer_entities
                }
                dataset.append(sample)
    except Exception as e:
        print(f"Error processing item: {item}")
        raise e

    finally:
        pat_train, pat_test = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=42)

        # Ensure data is saved even if an error occurs
        output_path = f'data/preprocessed/pat/pat_train.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': pat_train,
                'entities': list(all_entities),
                'relations': list(all_relations)
            }, f, ensure_ascii=False, indent=4)

        output_path = f'data/preprocessed/pat/pat_test.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': pat_test,
                'entities': list(all_entities),
                'relations': list(all_relations)
            }, f, ensure_ascii=False, indent=4)
        print(f"Data saved to {output_path}")

if __name__ == "__main__":
    print('Start')

    # asyncio.run(preprocess_lcquad('test'))
    # asyncio.run(preprocess_lcquad('train'))

    # asyncio.run(preprocess_pat())

    # asyncio.run(preprocess_rubq('train'))
    # asyncio.run(preprocess_rubq('test'))

    # asyncio.run(preprocess_qald('train'))
    # asyncio.run(preprocess_qald('test'))

