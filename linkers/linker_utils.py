import aiohttp
import asyncio
import requests

def extract_entity_info(response_json, entity_id):
    if entity_info := response_json['entities'].get(entity_id):
        # 1. Extract the label:
        label = None
        if labels := entity_info.get('labels'):
            if 'en' in labels:
                label = labels['en']['value']
            else:
                label = next(iter(labels.values()))['value']

        # 2. Extract the description:
        description = None
        if descriptions := entity_info.get('descriptions'):
            if 'en' in descriptions:
                description = descriptions['en']['value']
            else:
                description = next(iter(descriptions.values()))['value']

        # 3. Extract list of aliases:
        alias = []
        if aliases := entity_info.get('aliases'):
            if 'en' in aliases:
                alias = [alias_entry['value'] for alias_entry in aliases['en']]
            else:
                # Otherwise, compile aliases from all available languages
                for lang_aliases in aliases.values():
                    alias.extend([alias_entry['value'] for alias_entry in lang_aliases])

        return {
            'label': label,
            'description': description,
            'aliases': alias
        }

    else:
        raise ValueError(f'No information for entitiy {entity_id} found!')

def fetch_wikidata_entity_info(entity_id):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "format": "json",
        "props": "labels|descriptions|aliases",
        "languages": "en",
        "languagefallback": "1"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    return response.json()

async def fast_fetch_wikidata_entity_info(entity_id, session):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "format": "json",
        "props": "labels|descriptions|aliases",
        "languages": "en",
        "languagefallback": "1"
    }


    async with session.get(url, params=params) as response:
        response.raise_for_status()  # Raises an error for bad responses.
        response_json = await response.json()

        return response_json

async def fetch_multiple_entities(entity_ids, session):
    tasks = [asyncio.create_task(fast_fetch_wikidata_entity_info(eid, session)) for eid in entity_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful_results = {}
    failed_entities = {}

    for eid, result in zip(entity_ids, results):
        if isinstance(result, Exception):
            failed_entities[eid] = result
        else:
            successful_results[eid] = extract_entity_info(result, eid)

    return successful_results, failed_entities



