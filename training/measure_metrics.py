import re
import ssl
import json
import time
import pickle
import pandas as pd
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON
from preprocessing_utils import preprocess_sparql

ssl._create_default_https_context = ssl._create_unverified_context

def execute_sparql(query, timeout=60, max_retries=3):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    try:
        sparql.setQuery(query)
    except:
        return None
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(timeout)

    for attempt in range(1, max_retries + 1):
        try:
            results = sparql.query().convert()
            return extract_answers_from_response(results)
        except Exception as e:
            if 'QueryBadFormed' in str(e) or 'ExecutionException' in str(e):
                return None

            if attempt < max_retries:
                time.sleep(1)
            else:
                return []


def extract_answers_from_response(response):
    answers = []
    if 'results' in response:
        for binding in response['results']['bindings']:
            for key, sub_answer in binding.items():
                value = sub_answer.get('value')
                if re.match(r"^https?://www\.wikidata\.org/entity/Q\d+$", value):
                    answers.append(extract_wikidata_id_from_link(value))
                else:
                    answers.append(value)
    elif 'boolean' in response:
        answers.append(response['boolean'])
    return answers


def extract_wikidata_id_from_link(link):
    """
    Extract a Wikidata ID (e.g., Q12345) from a given link.
    """
    patterns = [
        r"https?://www\.wikidata\.org/entity/(Q\d+)",
        r"https?://www\.wikidata\.org/entity/(\d+)",
        r"https?://www\.wikidata\.org/wiki/(Q\d+)"
    ]
    for pattern in patterns:
        if match := re.search(pattern, link):
            return 'Q' + match.group(1) if pattern.endswith(r"/(\d+)") else match.group(1)
    return None


def fix_sparql_braces(query: str) -> str:
    open_count = query.count('{')
    close_count = query.count('}')

    if open_count == close_count:
        return query  # Query is already balanced.

    fixed_query = []  # List to collect characters for the new query.
    stack = []  # Stack to track unmatched '{'

    for char in query:
        if char == '{':
            fixed_query.append(char)
            stack.append('{')
        elif char == '}':
            if stack:
                fixed_query.append(char)
                stack.pop()
            else:
                # Found a closing '}' without a matching '{'
                fixed_query.append('{')  # Insert a missing opening brace.
                fixed_query.append(char)
        else:
            fixed_query.append(char)

    # If there are any unmatched '{' left, append the required number of '}' at the end.
    if stack:
        fixed_query.extend('}' * len(stack))

    return ''.join(fixed_query)


def calculate_metrics(correct, predicted):
    correct_set = set(correct)
    predicted_set = set(predicted)

    em = correct_set == predicted_set
    true_positives = len(correct_set & predicted_set)  # Intersection

    precision = true_positives / len(predicted_set) if predicted_set else 0
    recall = true_positives / len(correct_set) if correct_set else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {'em': em, 'f1': f1_score, 'precision': precision, 'recall': recall}

if __name__ == '__main__':
    dataset_name = 'rubq'
    path_to_preds_pkl = 'inference_results/rubq_inference_result.pkl'


    kgqa_test_dataset_list = json.load(open(f"../data/preprocessed/{dataset_name}/{dataset_name}_test.json"))['dataset']
    id2gold_query = {sample['id']: sample['query'] for sample in kgqa_test_dataset_list}

    preds_dict = pickle.load(open(path_to_preds_pkl, 'rb'))

    metrics_df = []
    entity_failed_pairs, predicate_failed_pairs = [], []

    print('Begin!')
    for id_, gold_query in tqdm(id2gold_query.items()):
        if str(id_) not in preds_dict:
            continue

        pred_query = preds_dict[str(id_)]['query']
        pred_query = fix_sparql_braces(pred_query)

        gold_entities = execute_sparql(gold_query)
        pred_entites = execute_sparql(pred_query)

        if not gold_entities:
            continue

        if pred_entites is None:
            metric = {'em': False, 'f1': 0, 'precision': 0.0, 'recall': 0, 'incorrect': True, 'empty': False}
        else:
            metric = calculate_metrics(gold_entities, pred_entites)
            metric.update({'incorrect': False, 'empty': len(pred_entites) == 0})

        metrics_df.append({'id': id_, **metric})

    metrics_df = pd.DataFrame(metrics_df).set_index('id')

    percentage_em = metrics_df["em"].mean() * 100  # Percentage of 'em' being True
    mean_f1 = metrics_df["f1"].mean()  # Mean F1-score
    percentage_incorrect = metrics_df["incorrect"].mean() * 100  # Percentage of 'incorrect' being True
    percentage_empty = metrics_df["empty"].mean() * 100  # Percentage of 'empty' being True

    # Display results
    metrics = {
        "Percentage of EM": f"{percentage_em:.2f}%",
        "Mean F1": f"{mean_f1:.2f}",
        "Mean Precision": f"{metrics_df["precision"].mean():.2f}",
        "Mean Recall": f"{metrics_df["recall"].mean():.2f}",
        "Percentage of Incorrect": f"{percentage_incorrect:.2f}%",
        "Percentage of Empty": f"{percentage_empty:.2f}%"
    }

    # Print metrics
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print()
    print(metrics_df)