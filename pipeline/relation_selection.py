import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
from typing import Dict
import random
import time
from utils.model_utils import load_model
from utils.data_processing import (
    parse_xml_file, extract_entities_and_types,
    find_entity_info, find_relation_labels,
    get_example_candidate_relations, get_triples,
    generate_output, get_seed_from_filename
)
from utils.prompt_utils import create_rs_prompt


def run_relation_selection(config: Dict, api_config: Dict = None, et_output_folder: str = None) -> str:
    """Run relation selection with resume capability"""
    # Load model
    model_wrapper = load_model(
        {
            'name': config['models']['relation_selection']['name'],
            'params': config['models']['relation_selection']['params']
        },
        api_config
    )

    # Prepare paths
    xml_folder = config['paths']['train']
    ontology_folder = config['paths']['ontology']
    output_folder = config['output']['pipeline']['relation_selection']
    hierarchy_xml = config['paths']['hierarchy_xml']
    os.makedirs(output_folder, exist_ok=True)

    # Parse hierarchy XML
    entity_paths = parse_xml_file(hierarchy_xml)

    # Initialize progress bar
    json_files = [f for f in os.listdir(et_output_folder) if f.endswith('.json')]
    total_entries = sum(len(json.load(open(os.path.join(et_output_folder, f), 'r', encoding='utf-8')))
                        for f in json_files)
    pbar = tqdm(total=total_entries, desc="Processing entries")

    for json_file in json_files:
        json_path = os.path.join(et_output_folder, json_file)
        xml_path = os.path.join(xml_folder, json_file.replace('.json', '.xml'))
        output_path = os.path.join(output_folder, json_file)

        # Check for existing results
        processed_ids = set()
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    processed_ids = {item['id'] for item in existing_data}
            except json.JSONDecodeError:
                existing_data = []
                processed_ids = set()
        else:
            existing_data = []

        if not os.path.exists(xml_path):
            print(f"Warning: XML file {xml_path} not found")
            pbar.update(len(existing_data))
            continue

        # Load XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Process each entry
        results = existing_data.copy()
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for entry in data:
            entry_id = entry['id']

            # Skip already processed entries
            if entry_id in processed_ids:
                pbar.update(1)
                continue

            try:
                text = entry['sent']
                response = entry['response']

                # Select example based on method
                if config['general']['example_selection'] == "example":
                    example_file_path = os.path.join(config['paths']['example'], json_file)
                    if not os.path.exists(example_file_path):
                        print(f"Warning: Example file {example_file_path} not found")
                        pbar.update(1)
                        continue

                    with open(example_file_path, 'r', encoding='utf-8') as ef:
                        example_data = json.load(ef)

                    if entry_id not in example_data:
                        print(f"Warning: No example entry found for {entry_id}")
                        pbar.update(1)
                        continue

                    example_info = example_data[entry_id]["Rank_1"]
                    example_train_id = example_info["Train_ID"]
                    example_entry = root.find(f".//entry[@id='{example_train_id}']")
                    if not example_entry:
                        print(f"Warning: Example entry {example_train_id} not found")
                        pbar.update(1)
                        continue

                    example_text = example_entry.find("text").text.replace("_", " ")

                    example_entities_set = set()
                    for triple in example_entry.find("triples").findall("triple"):
                        sub = triple.find("sub").text.replace("_", " ")
                        obj = triple.find("obj").text.replace("_", " ")
                        example_entities_set.update([f"{{{sub}}}", f"{{{obj}}}"])
                    example_entities = ", ".join(example_entities_set)

                    example_output_list = [
                        f"[{triple.find('rel').text.replace('_', ' ')}]{{{triple.find('sub').text.replace('_', ' ')}, {triple.find('obj').text.replace('_', ' ')}}}"
                        for triple in example_entry.find("triples").findall("triple")
                    ]
                    example_output = ", ".join(example_output_list)

                    example_candidate_relations = get_example_candidate_relations(
                        example_entry, entity_paths, ontology_folder
                    )
                else:
                    # Random selection with fixed seed
                    seed = get_seed_from_filename(json_file)
                    random.seed(seed)
                    train_entries = [e for e in root.findall('.//entry') if 'train' in e.attrib['id']]
                    if not train_entries:
                        print(f"Warning: No train entries in {xml_path}")
                        pbar.update(1)
                        continue

                    example_entry = random.choice(train_entries)
                    example_id = example_entry.attrib['id']
                    example_text = example_entry.find("text").text.replace("_", " ")

                    example_entities_set = set()
                    for triple in example_entry.find("triples").findall("triple"):
                        sub = triple.find("sub").text.replace("_", " ")
                        obj = triple.find("obj").text.replace("_", " ")
                        example_entities_set.update([f"{{{sub}}}", f"{{{obj}}}"])
                    example_entities = ", ".join(example_entities_set)

                    example_output_list = [
                        f"[{triple.find('rel').text.replace('_', ' ')}]{{{triple.find('sub').text.replace('_', ' ')}, {triple.find('obj').text.replace('_', ' ')}}}"
                        for triple in example_entry.find("triples").findall("triple")
                    ]
                    example_output = ", ".join(example_output_list)

                    example_candidate_relations = get_example_candidate_relations(
                        example_entry, entity_paths, ontology_folder
                    )

                # Extract entities
                entities = ", ".join(
                    [f"{{{entity.split('}')[0]}}}" for entity in response.split(", ")]
                )

                # Find candidate relations
                try:
                    response_entities = extract_entities_and_types(response)
                    entity_types = set()

                    for entity, entity_type in response_entities.items():
                        path, path_to_leaves = find_entity_info(entity_type, entity_paths)
                        entity_types.update(path)
                        for leaf_path in path_to_leaves:
                            entity_types.update(tuple(leaf_path))

                    ontology_file_name = f"{'_'.join(entry_id.split('_')[:2])}.json"
                    ontology_file_path = os.path.join(ontology_folder, ontology_file_name)

                    if os.path.exists(ontology_file_path):
                        relations = find_relation_labels(ontology_file_path, entity_types)
                        relations = ", ".join([f"[{rel}]" for rel in relations])
                    else:
                        relations = ""
                except Exception as e:
                    print(f"Error processing relations for {entry_id}: {e}")
                    relations = ""

                # Create prompt
                prompt = create_rs_prompt(
                    text, entities, relations,
                    example_text, example_entities, example_output,
                    example_candidate_relations
                )

                # Generate response with retry
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = model_wrapper.generate_response(prompt)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        print(f"Attempt {attempt + 1} failed for {entry_id}: {str(e)}")
                        time.sleep(5 * (attempt + 1))

                # Store result
                results.append({
                    "id": entry_id,
                    "sent": text,
                    "response": response
                })

                # Save progress every 5 entries
                if len(results) % 5 == 0:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=4)

                pbar.update(1)

            except Exception as e:
                print(f"Error processing entry {entry_id}: {str(e)}")
                # Save current progress
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                continue

        # Save final results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    pbar.close()
    return output_folder
