import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
from typing import Dict
import random
import time
from utils.model_utils import load_model
from utils.data_processing import (
    parse_xml_file, get_candidate_types, extract_entities_and_types,
    find_entity_info, get_triples, generate_output, get_seed_from_filename
)
from utils.prompt_utils import create_et_prompt


def run_entity_typing(config: Dict, api_config: Dict = None, er_output_folder: str = None) -> str:
    """Run entity typing with resume capability"""
    # Load model
    model_wrapper = load_model(
        {
            'name': config['models']['entity_typing']['name'],
            'params': config['models']['entity_typing']['params']
        },
        api_config
    )

    # Prepare paths
    xml_folder = config['paths']['train']
    ontology_folder = config['paths']['ontology']
    output_folder = config['output']['pipeline']['entity_typing']
    hierarchy_xml = config['paths']['hierarchy_xml']
    os.makedirs(output_folder, exist_ok=True)

    # Parse hierarchy XML
    entity_paths = parse_xml_file(hierarchy_xml)

    # Initialize progress bar
    json_files = [f for f in os.listdir(er_output_folder) if f.endswith('.json')]
    total_entries = sum(len(json.load(open(os.path.join(er_output_folder, f), 'r', encoding='utf-8')))
                        for f in json_files)
    pbar = tqdm(total=total_entries, desc="Processing entries")

    prompts = []

    for json_file in json_files:
        json_path = os.path.join(er_output_folder, json_file)
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

        # Get candidate types
        candidate_types = get_candidate_types(json_file, ontology_folder, hierarchy_xml)

        # Process each entry
        results = existing_data.copy()
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            item_id = item['id']

            # Skip already processed entries
            if item_id in processed_ids:
                pbar.update(1)
                continue

            try:
                text = item['sent']
                entities_str = item['response']

                # Select example based on method
                if config['general']['example_selection'] == "example":
                    example_file_path = os.path.join(config['paths']['example'], json_file)
                    if not os.path.exists(example_file_path):
                        print(f"Warning: Example file {example_file_path} not found")
                        pbar.update(1)
                        continue

                    with open(example_file_path, 'r', encoding='utf-8') as ef:
                        example_data = json.load(ef)

                    if item_id not in example_data or "Rank_1" not in example_data[item_id]:
                        print(f"Warning: No Rank_1 example for {item_id}")
                        pbar.update(1)
                        continue

                    example_train_id = example_data[item_id]["Rank_1"]["Train_ID"]
                    example_entry = root.find(f".//entry[@id='{example_train_id}']")
                    if not example_entry:
                        print(f"Warning: Example entry {example_train_id} not found")
                        pbar.update(1)
                        continue

                    example_text = example_entry.find('text').text
                    example_triples = get_triples(example_entry, replace_underscore=True)
                    example_output = generate_output(example_entry, replace_underscore=True)
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
                    example_text = example_entry.find('text').text
                    example_triples = get_triples(example_entry, replace_underscore=True)
                    example_output = generate_output(example_entry, replace_underscore=True)

                # Create prompt
                prompt = create_et_prompt(
                    text, entities_str, candidate_types,
                    example_text, example_triples, example_output
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
                        print(f"Attempt {attempt + 1} failed for {item_id}: {str(e)}")
                        time.sleep(5 * (attempt + 1))

                # Store result
                results.append({
                    "id": item_id,
                    "sent": text,
                    "response": response
                })

                # Save progress every 5 entries
                if len(results) % 5 == 0:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=4)

                pbar.update(1)

            except Exception as e:
                print(f"Error processing entry {item_id}: {str(e)}")
                # Save current progress
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                continue

        # Save final results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    pbar.close()
    return output_folder
