import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
from typing import Dict, Tuple, List
import random
import time
from utils.model_utils import load_model
from utils.data_processing import (
    get_seed_from_filename, clean_and_deduplicate_entities
)
from utils.prompt_utils import create_er_prompt

def select_example_from_train_file(train_filepath: str, train_id: str = None) -> Tuple[str, str, str]:
    """Select example from train file based on selection method."""
    tree = ET.parse(train_filepath)
    root = tree.getroot()
    entries = root.find('entries')

    if train_id:
        # Example selection method
        example_entry = next((e for e in entries.findall('entry') if e.attrib['id'] == train_id), None)
        if not example_entry:
            raise ValueError(f"Train entry with ID {train_id} not found in file: {train_filepath}")
    else:
        # Random selection method
        all_entries = [e for e in entries.findall('entry') if 'train' in e.attrib['id']]
        if not all_entries:
            raise ValueError(f"No 'train' entries found in the train file: {train_filepath}")
        seed = get_seed_from_filename(train_filepath)
        random.seed(seed)
        example_entry = random.choice(all_entries)
        train_id = example_entry.attrib['id']

    example_text = example_entry.find('text').text
    triples = example_entry.find('triples').findall('triple')

    entities = set()
    for triple in triples:
        entities.add(triple.find('sub').text)
        entities.add(triple.find('obj').text)

    cleaned_entities = clean_and_deduplicate_entities(entities)
    example_output = ', '.join([f"[{entity}]" for entity in cleaned_entities])

    return example_text, example_output, train_id


def run_entity_recognition(config: Dict, api_config: Dict = None) -> str:
    """Run entity recognition with resume capability"""
    # Load model
    model_wrapper = load_model(
        {
            'name': config['models']['entity_recognition']['name'],
            'params': config['models']['entity_recognition']['params']
        },
        api_config
    )

    # Prepare paths
    test_folder = config['paths']['test']
    train_folder = config['paths']['train']
    output_folder = config['output']['pipeline']['entity_recognition']
    os.makedirs(output_folder, exist_ok=True)

    # Get test files
    test_files = [f for f in os.listdir(test_folder) if f.endswith('.xml')]

    # Initialize progress bar
    total_entries = sum(len(ET.parse(os.path.join(test_folder, f)).getroot().find('entries'))
                        for f in test_files)
    pbar = tqdm(total=total_entries, desc="Processing test entries")

    example_data = {}
    if config['general']['example_selection'] == "example":
        example_folder = config['paths']['example']
        example_files = {os.path.splitext(f)[0]: os.path.join(example_folder, f)
                         for f in os.listdir(example_folder)}

    for test_file in test_files:
        test_filepath = os.path.join(test_folder, test_file)
        train_filepath = os.path.join(train_folder, test_file)
        output_file_path = os.path.join(output_folder, test_file.replace('.xml', '.json'))

        # Check for existing results
        processed_ids = set()
        if os.path.exists(output_file_path):
            try:
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    processed_ids = {item['id'] for item in existing_data}
            except json.JSONDecodeError:
                existing_data = []
                processed_ids = set()
        else:
            existing_data = []

        if not os.path.exists(train_filepath):
            print(f"Warning: Corresponding train file not found for {test_file}")
            continue

        # Load example data if using example selection
        if config['general']['example_selection'] == "example":
            example_key = os.path.splitext(test_file)[0]
            if example_key not in example_files:
                print(f"Warning: Example file for {test_file} not found")
                continue
            with open(example_files[example_key], 'r', encoding='utf-8') as ef:
                example_data = json.load(ef)

        # Parse test file
        tree = ET.parse(test_filepath)
        root = tree.getroot()
        file_output = existing_data.copy()

        # Process each entry
        for entry in root.find('entries').findall('entry'):
            entry_id = entry.attrib['id']

            # Skip already processed entries
            if entry_id in processed_ids:
                pbar.update(1)
                continue

            try:
                text = entry.find('text').text

                # Get corresponding example
                if config['general']['example_selection'] == "example":
                    if entry_id not in example_data or "Rank_1" not in example_data[entry_id]:
                        print(f"Warning: No Rank_1 example for {entry_id}")
                        pbar.update(1)
                        continue

                    train_id = example_data[entry_id]["Rank_1"]["Train_ID"]
                    example_text, example_output, selected_id = select_example_from_train_file(
                        train_filepath, train_id
                    )
                else:
                    # Random selection with fixed seed
                    seed = get_seed_from_filename(test_file)
                    random.seed(seed)
                    example_text, example_output, selected_id = select_example_from_train_file(
                        train_filepath
                    )

                # Create prompt and get response
                prompt = create_er_prompt(text, example_text, example_output)

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
                file_output.append({
                    "id": entry_id,
                    "sent": text,
                    "response": response
                })

                # Save progress every 5 entries
                if len(file_output) % 5 == 0:
                    with open(output_file_path, 'w', encoding='utf-8') as f:
                        json.dump(file_output, f, ensure_ascii=False, indent=4)

                pbar.update(1)

            except Exception as e:
                print(f"Error processing entry {entry_id}: {str(e)}")
                # Save current progress
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(file_output, f, ensure_ascii=False, indent=4)
                continue

        # Save final results for this file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(file_output, f, ensure_ascii=False, indent=4)

    pbar.close()
    return output_folder