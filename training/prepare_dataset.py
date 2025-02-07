import json

def create_prompt(item):
    question = item['en_question']
    return f'''Question: {}\nEntities:{}'''


dataset_name = 'pat'

train_data = json.load(open(f"../clean_data/preprocessed/{dataset_name}/{dataset_name}_train.json"))
test_data = json.load(open(f"../clean_data/preprocessed/{dataset_name}/{dataset_name}_test.json"))