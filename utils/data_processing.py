import xml.etree.ElementTree as ET
import json
import re
import os
import hashlib
from typing import List, Dict, Tuple, Set

def parse_xml_file(xml_file_path: str) -> Dict:
    """Parse XML file and extract entity paths and hierarchies."""
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    entity_paths = {}

    def find_entity_paths(node, path):
        current_path = path.copy()
        if 'name' in node.attrib:
            entity_type = node.attrib['name']
            current_path.append(entity_type)
            entity_paths[entity_type] = {
                "path_to_root": current_path.copy(),
                "path_to_leaves": set()
            }

            def find_leaf_paths(current_node, current_leaf_path):
                current_leaf_path = current_leaf_path.copy()
                if 'name' in current_node.attrib:
                    current_leaf_path.append(current_node.attrib['name'])
                if len(current_node) == 0:
                    entity_paths[entity_type]['path_to_leaves'].add(tuple(current_leaf_path))
                for child in current_node:
                    find_leaf_paths(child, current_leaf_path)

            find_leaf_paths(node, current_path)

        for child in node:
            find_entity_paths(child, current_path)

    find_entity_paths(root, [])
    for key in entity_paths:
        entity_paths[key]["path_to_leaves"] = [list(leaf) for leaf in entity_paths[key]["path_to_leaves"]]
    return entity_paths

def extract_entities_and_types(response_str: str) -> Dict[str, str]:
    """Extract entities and their types from response string."""
    pattern = r'\{(.+?)\}:\[(.+?)\]'
    matches = re.findall(pattern, response_str)
    return {entity: entity_type for entity, entity_type in matches}

def get_seed_from_filename(filename: str) -> int:
    """Generate consistent seed from filename."""
    return int(hashlib.md5(filename.encode()).hexdigest(), 16)

def clean_and_deduplicate_entities(entities: List[str]) -> Set[str]:
    """Clean entities by replacing underscores and deduplicate."""
    cleaned_entities = set()
    for entity in entities:
        cleaned_entities.add(entity.replace('_', ' '))
    return cleaned_entities

def parse_ontology_schema(ontology_file: str) -> str:
    """Parse ontology schema file and format it."""
    with open(ontology_file, 'r', encoding='utf-8') as f:
        ontology_data = json.load(f)

    entity_type = ", ".join([et['label'] for et in ontology_data.get('entity type', [])])
    relation = ", ".join([f"({rel['domain']}, {rel['label']}, {rel['range']})" for rel in ontology_data.get('relation', [])])
    hierarchy = ", ".join([f"({hier['domain']}, {hier['label']}, {hier['range']})" for hier in ontology_data.get('hierarchy', [])])

    ontology_schema = (
        f"entity type: {entity_type}\n"
        f"relation: {relation}\n"
        f"hierarchy: {hierarchy}"
    )
    return ontology_schema

def get_candidate_types(json_filename: str, ontology_folder: str, hierarchy_xml: str) -> str:
    """Get candidate entity types from ontology and hierarchy."""
    original_tree = ET.parse(hierarchy_xml)
    original_root = original_tree.getroot()

    json_path = os.path.join(ontology_folder, json_filename)

    with open(json_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        relevant_entities = {entity['label'] for entity in json_data.get("entity type", [])}

    def trim_xml_tree(root, relevant_entities):
        def find_paths_to_root(entity_name, path):
            for element in root.findall(".//entity_type[@name='" + entity_name + "']"):
                current_path = [element] + path
                parent = element.find("..")
                if parent is not None and parent.tag == "entity_type":
                    yield from find_paths_to_root(parent.attrib['name'], current_path)
                else:
                    yield current_path

        elements_to_keep = set()
        for entity in relevant_entities:
            for path in find_paths_to_root(entity, []):
                elements_to_keep.update(path)

        new_root = ET.Element(root.tag, root.attrib)

        def copy_element_with_children(element, new_parent):
            new_element = ET.SubElement(new_parent, element.tag, element.attrib)
            for child in element:
                if child in elements_to_keep:
                    copy_element_with_children(child, new_element)

        for element in root:
            if element in elements_to_keep:
                copy_element_with_children(element, new_root)

        return ET.ElementTree(new_root)

    trimmed_tree = trim_xml_tree(original_root, relevant_entities)

    def parse_entity(element):
        text = f"[{element.attrib['name']}]"
        children = list(element)
        if children:
            child_texts = [parse_entity(child) for child in children]
            text += ":{ " + ", ".join(child_texts) + " }"
        return text

    result = []
    for entity in trimmed_tree.getroot().findall('entity_type'):
        result.append(parse_entity(entity))

    def get_xml_entity_types(root):
        entities = set()
        for entity in root.findall(".//entity_type"):
            entities.add(entity.attrib['name'])
        return entities

    trimmed_entity_types = get_xml_entity_types(trimmed_tree.getroot())
    missing_entities = []
    with open(json_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        for entity in json_data.get("entity type", []):
            if entity['label'] not in trimmed_entity_types:
                missing_entities.append(entity['label'])

    if missing_entities:
        for missing_entity in missing_entities:
            result.append(f"[{missing_entity}]")

    return ", ".join(result)

def find_entity_info(entity_type: str, entity_paths: Dict) -> Tuple[List[str], List[List[str]]]:
    """Find entity info from pre-parsed entity paths."""
    if entity_type in entity_paths:
        entity_info = entity_paths[entity_type]
        path_to_root = entity_info['path_to_root']
        path_to_leaves = entity_info['path_to_leaves']
        return path_to_root, path_to_leaves
    return [entity_type], []

def find_relation_labels(ontology_json_file: str, entity_types: Set[str]) -> List[str]:
    """Find relation labels from ontology that match given entity types."""
    with open(ontology_json_file, 'r', encoding='utf-8') as f:
        ontology_data = json.load(f)

    relations = ontology_data.get("relation", [])
    labels = set()

    for relation in relations:
        if relation['domain'] in entity_types and relation['range'] in entity_types:
            labels.add(relation['label'])

    return list(labels)

def get_example_candidate_relations(example_entry: ET.Element, entity_paths: Dict, ontology_folder: str) -> str:
    """Get candidate relations for example entry."""
    example_schemas = example_entry.find("schemas")
    if example_schemas is None:
        return ""

    candidate_relations = set()

    for schema in example_schemas.findall("schema"):
        sub_type = schema.find("sub").text
        obj_type = schema.find("obj").text

        sub_path, _ = find_entity_info(sub_type, entity_paths)
        obj_path, _ = find_entity_info(obj_type, entity_paths)

        category = example_entry.attrib["category"]
        ontology_file_name = f"{category}.json"
        ontology_file_path = os.path.join(ontology_folder, ontology_file_name)

        if os.path.exists(ontology_file_path):
            entity_types = set(sub_path + obj_path)
            relations = find_relation_labels(ontology_file_path, entity_types)
            candidate_relations.update(relations)

    return ", ".join([f"[{rel}]" for rel in candidate_relations])

def get_triples(entry: ET.Element, replace_underscore: bool = False) -> str:
    """Get triples from XML entry and optionally replace underscores."""
    entities = set()
    for triple in entry.findall('.//triple'):
        sub = triple.find('sub').text
        obj = triple.find('obj').text
        if replace_underscore:
            sub = sub.replace('_', ' ')
            obj = obj.replace('_', ' ')
        entities.add(f'[{sub}]')
        entities.add(f'[{obj}]')
    return ', '.join(sorted(entities))

def generate_output(entry: ET.Element, replace_underscore: bool = False) -> str:
    """Generate formatted output from XML entry."""
    entity_output = []
    triples = entry.findall('.//triple')
    schemas = entry.findall('.//schema')

    entity_dict = {}
    for triple, schema in zip(triples, schemas):
        sub = triple.find('sub').text
        obj = triple.find('obj').text
        sub_type = schema.find('sub').text
        obj_type = schema.find('obj').text

        if replace_underscore:
            sub = sub.replace('_', ' ')
            obj = obj.replace('_', ' ')

        entity_dict[sub] = sub_type
        entity_dict[obj] = obj_type

    for entity, entity_type in entity_dict.items():
        entity_output.append(f'{{{entity}}}:[{entity_type}]')
    return ', '.join(entity_output)


def select_example_from_json(example_filepath: str, test_entry_id: str) -> str:
    """Select example from JSON file based on test entry ID.

    Args:
        example_filepath: Path to the example JSON file
        test_entry_id: ID of the test entry to find example for

    Returns:
        The train ID of the selected example

    Raises:
        ValueError: If test entry ID not found or no Rank_1 data
    """
    with open(example_filepath, 'r', encoding='utf-8') as f:
        example_data = json.load(f)

    if test_entry_id not in example_data:
        raise ValueError(f"Test ID {test_entry_id} not found in example file: {example_filepath}")

    rank_1_data = example_data[test_entry_id].get("Rank_1")
    if not rank_1_data:
        raise ValueError(f"Rank_1 data not found for Test ID {test_entry_id} in file: {example_filepath}")

    return rank_1_data["Train_ID"]


def get_train_data_by_id(train_filepath: str, train_id: str) -> Tuple[str, str]:
    """Get train text and output by entry ID.

    Args:
        train_filepath: Path to the train XML file
        train_id: ID of the train entry to retrieve

    Returns:
        Tuple of (example_text, example_output)

    Raises:
        ValueError: If train ID not found
    """
    tree = ET.parse(train_filepath)
    root = tree.getroot()
    entries = root.find('entries')

    for entry in entries.findall('entry'):
        if entry.attrib['id'] == train_id:
            example_text = entry.find('text').text

            triples = entry.find('triples').findall('triple')
            schemas = entry.find('schemas').findall('schema')

            example_output = ', '.join([
                f"[{triple.find('sub').text.replace('_', ' ')}, {triple.find('rel').text}, {triple.find('obj').text.replace('_', ' ')}]:"
                f"({schema.find('sub').text}, {schema.find('rel').text}, {schema.find('obj').text})"
                for triple, schema in zip(triples, schemas)
            ])

            return example_text, example_output

    raise ValueError(f"Train ID {train_id} not found in train file: {train_filepath}")