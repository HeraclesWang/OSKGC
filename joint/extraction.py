import os
import json
import xml.etree.ElementTree as ET
import random
from tqdm import tqdm
from typing import Dict
from utils.model_utils import load_model
from utils.data_processing import (
    parse_ontology_schema, get_seed_from_filename,
    select_example_from_json, get_train_data_by_id
)
from utils.prompt_utils import create_joint_prompt
import time


def run_joint_extraction(config: Dict, api_config: Dict = None):
    """Run joint extraction approach with resume capability"""
    # Load model
    model_wrapper = load_model(
        {
            'name': config['models']['joint_extraction']['name'],
            'params': config['models']['joint_extraction']['params']
        },
        api_config
    )

    # Prepare paths
    test_folder = config['paths']['test']
    train_folder = config['paths']['train']
    ontology_folder = config['paths']['ontology']
    output_folder = config['output']['joint']
    os.makedirs(output_folder, exist_ok=True)

    # Get test files
    test_files = [f for f in os.listdir(test_folder) if f.endswith('.xml')]
    ontology_files = {os.path.splitext(f)[0]: os.path.join(ontology_folder, f)
                      for f in os.listdir(ontology_folder)}

    if config['general']['example_selection'] == "example":
        example_folder = config['paths']['example']
        example_files = {os.path.splitext(f)[0]: os.path.join(example_folder, f)
                         for f in os.listdir(example_folder)}

    # Initialize progress bar
    total_entries = sum(len(ET.parse(os.path.join(test_folder, f)).getroot().find('entries'))
                        for f in test_files)
    pbar = tqdm(total=total_entries, desc="Processing test entries")

    for test_file in test_files:
        test_filepath = os.path.join(test_folder, test_file)
        train_filepath = os.path.join(train_folder, test_file)
        output_file_path = os.path.join(output_folder, f"{test_file.replace('.xml', '.json')}")

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

        ontology_title = os.path.splitext(test_file)[0]
        ontology_filepath = ontology_files.get(ontology_title)
        if not ontology_filepath:
            print(f"Warning: Corresponding ontology file not found for {test_file}")
            continue

        # Parse ontology file
        ontology_schema = parse_ontology_schema(ontology_filepath)

        # Select example based on selection method
        if config['general']['example_selection'] == "example":
            example_filepath = example_files.get(ontology_title)
            if not example_filepath:
                print(f"Warning: Example file not found for {test_file}")
                continue

        # Parse test file
        tree = ET.parse(test_filepath)
        root = tree.getroot()

        # Store results for current file
        file_output = existing_data.copy()

        # Process each entry
        entries = root.find('entries')
        for entry in entries.findall('entry'):
            entry_id = entry.attrib['id']

            # Skip already processed entries
            if entry_id in processed_ids:
                pbar.update(1)
                continue

            try:
                # Select example
                if config['general']['example_selection'] == "example":
                    try:
                        train_id = select_example_from_json(example_filepath, entry_id)
                        example_text, example_output = get_train_data_by_id(train_filepath, train_id)
                    except ValueError as e:
                        print(f"Skipping entry {entry_id}: {str(e)}")
                        pbar.update(1)
                        continue
                else:
                    # Random selection with fixed seed
                    seed = get_seed_from_filename(test_file)
                    random.seed(seed)
                    train_entries = [e for e in ET.parse(train_filepath).getroot().find('entries').findall('entry')
                                     if 'train' in e.attrib['id']]
                    if not train_entries:
                        print(f"Warning: No train entries in {train_filepath}")
                        pbar.update(1)
                        continue

                    example_entry = random.choice(train_entries)
                    example_text = example_entry.find('text').text
                    triples = example_entry.find('triples').findall('triple')
                    schemas = example_entry.find('schemas').findall('schema')

                    example_output = ', '.join([
                        f"[{triple.find('sub').text.replace('_', ' ')}, {triple.find('rel').text}, {triple.find('obj').text.replace('_', ' ')}]:"
                        f"({schema.find('sub').text}, {schema.find('rel').text}, {schema.find('obj').text})"
                        for triple, schema in zip(triples, schemas)
                    ])

                # Create prompt
                input_text = entry.find('text').text
                prompt = create_joint_prompt(input_text, example_text, example_output, ontology_schema)

                # Generate response with retry mechanism
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = model_wrapper.generate_response(prompt)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        print(f"Attempt {attempt + 1} failed for {entry_id}: {str(e)}")
                        time.sleep(5 * (attempt + 1))  # Exponential backoff

                # Append result
                file_output.append({
                    "id": entry_id,
                    "input_text": input_text,
                    "response": response
                })

                # Save progress every 5 entries
                if len(file_output) % 5 == 0:
                    with open(output_file_path, 'w', encoding='utf-8') as out_f:
                        json.dump(file_output, out_f, ensure_ascii=False, indent=4)

                pbar.update(1)

            except Exception as e:
                print(f"Error processing entry {entry_id}: {str(e)}")
                # Save whatever we have so far
                with open(output_file_path, 'w', encoding='utf-8') as out_f:
                    json.dump(file_output, out_f, ensure_ascii=False, indent=4)
                continue

        # Save final results for this file
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            json.dump(file_output, out_f, ensure_ascii=False, indent=4)

    pbar.close()