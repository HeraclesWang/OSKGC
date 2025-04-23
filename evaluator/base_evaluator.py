import os
import json
import xml.etree.ElementTree as ET
import re
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class BaseEvaluator:
    def __init__(self, config: Dict):
        self.config = config
        eval_dir_name = f"{config['model_name']}_{config['general']['approach']}_{config['general']['example_selection']}"
        self.output_dir = os.path.join(f"{config['evaluation']['output_dir']}", eval_dir_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.detailed_dir = os.path.join(self.output_dir, "detailed_results")
        os.makedirs(self.detailed_dir, exist_ok=True)

        self.triple_eval_file = os.path.join(self.output_dir, "triple_evaluation.jsonl")
        self.overall_eval_file = os.path.join(self.output_dir, "overall_evaluation.jsonl")
        self.ss_eval_file = os.path.join(self.output_dir, "SS.jsonl")

    def parse_xml(self, xml_path: str) -> Dict[str, List[Tuple[str, str, str]]]:
        """Parse XML file to extract triples by entry ID."""
        try:
            with open(xml_path, 'r', encoding='utf-8') as f:
                tree = ET.parse(f)
        except (ET.ParseError, UnicodeDecodeError) as e:
            print(f"Error parsing XML file {xml_path}: {str(e)}")
            return {}
        root = tree.getroot()

        data = {}
        for entry in root.find("entries"):
            entry_id = entry.get("id")
            triples = []
            for triple in entry.find("triples"):
                sub = triple.find("sub").text.replace("_", " ").lower()
                rel = triple.find("rel").text.lower()
                obj = triple.find("obj").text.replace("_", " ").lower()
                triples.append((sub, rel, obj))
            data[entry_id] = triples
        return data

    def calculate_metrics(self, predicted: List[Tuple], actual: List[Tuple]) -> Tuple[float, float, float]:
        """Calculate precision, recall and F1 score."""
        predicted_set = set(predicted)
        actual_set = set(actual)

        true_positives = len(predicted_set & actual_set)
        precision = true_positives / len(predicted_set) if predicted_set else 0
        recall = true_positives / len(actual_set) if actual_set else 0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0

        return round(precision, 3), round(recall, 3), round(f1, 3)

    def parse_hierarchy_file(self) -> Tuple[Dict, Dict, Dict]:
        """Parse hierarchy XML file."""
        tree = ET.parse(self.config['paths']['hierarchy_xml'])
        root = tree.getroot()
        hierarchy = {}
        child_counts = {}
        sibling_counts = {}

        def build_hierarchy(node, path):
            name = node.get("name")
            if not name:
                return
            hierarchy[name] = path + [name]
            children = node.findall("entity_type")
            child_counts[name] = len(children)

            for child in children:
                child_name = child.get("name")
                if child_name:
                    sibling_counts[child_name] = len(children) - 1

            for child in children:
                build_hierarchy(child, path + [name])

        for child in root.findall("entity_type"):
            build_hierarchy(child, [])

        return hierarchy, child_counts, sibling_counts

    def calculate_ss_score(self, hierarchy: Dict, child_counts: Dict, sibling_counts: Dict,
                           true_type: str, pred_type: str) -> Tuple:
        """Calculate structural similarity score."""
        if true_type == pred_type:
            return (self.get_distance_to_root(hierarchy, true_type), 0, 1)

        if true_type not in hierarchy or pred_type not in hierarchy:
            return (None, None, 0)

        D = self.get_distance_to_root(hierarchy, true_type)
        if D == 0:
            return ('root', 'root', 1 if true_type == pred_type else 0)

        true_path = hierarchy[true_type]
        if pred_type in true_path:
            d = len(true_path) - true_path.index(pred_type) - 1
            alpha = 2
            score = math.exp(-alpha * d / D)
            return (D, d, score)

        distance, lca = self.get_distance_between_types(hierarchy, true_type, pred_type)
        if not lca:
            return (D, None, 0)

        d = len(hierarchy[true_type]) - hierarchy[true_type].index(lca) - 1
        d_prime = len(hierarchy[pred_type]) - hierarchy[pred_type].index(lca) - 1

        S = sibling_counts.get(pred_type, 0)
        entropy = math.log2(S + 1)

        alpha = 2
        beta = 1.5
        score = math.exp(-alpha * d / D) * math.exp(-beta * d_prime / (D + entropy))

        return (D, d, score)

    def get_distance_between_types(self, hierarchy: Dict, type1: str, type2: str) -> Tuple:
        """Calculate distance between two types in hierarchy."""
        if type1 in hierarchy and type2 in hierarchy:
            path1 = hierarchy[type1]
            path2 = hierarchy[type2]

            common_ancestor = None
            for t, p in zip(path1, path2):
                if t == p:
                    common_ancestor = t
                else:
                    break

            if common_ancestor:
                distance = (len(path1) - path1.index(common_ancestor) - 1) + \
                           (len(path2) - path2.index(common_ancestor) - 1)
                return distance, common_ancestor
        return None, None

    def get_distance_to_root(self, hierarchy: Dict, entity_type: str) -> Optional[int]:
        """Calculate distance from entity type to root."""
        if entity_type in hierarchy:
            return len(hierarchy[entity_type]) - 1
        return None


    def process_structural_similarity(self) -> Tuple[Dict, float]:
        """Evaluate structural similarity."""
        schema_data = self.process_schema_stage()
        hierarchy, child_counts, sibling_counts = self.parse_hierarchy_file()
        ss_results = []

        for record in schema_data:
            result = self.process_record(record, hierarchy, child_counts, sibling_counts)
            if result:
                ss_results.append(result)

        # Save SS results
        self.save_results(ss_results, self.ss_eval_file)

        # Calculate average scores by prefix and overall
        prefix_scores, overall_avg = self.calculate_scores_from_ss(ss_results)
        return prefix_scores, overall_avg


    def process_file(self, json_path: str, xml_data: Dict) -> Tuple:
        """Process a single JSON file and calculate metrics."""
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        results = []
        all_predicted = []
        all_actual = []
        per_record_f1 = []

        for item in json_data:
            entry_id = item["id"]
            response = item.get("response", "")
            predicted = self.parse_response(response)
            actual = xml_data.get(entry_id, [])

            precision, recall, f1 = self.calculate_metrics(predicted, actual)

            results.append({
                "id": entry_id,
                "response": response,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

            all_predicted.extend(predicted)
            all_actual.extend(actual)
            per_record_f1.append(f1)

        overall_precision, overall_recall, overall_f1 = self.calculate_metrics(all_predicted, all_actual)
        macro_f1 = round(sum(per_record_f1) / len(per_record_f1), 3) if per_record_f1 else 0

        return results, overall_precision, overall_recall, overall_f1, macro_f1


    def process_record(self, record: Dict, hierarchy: Dict,
                      child_counts: Dict, sibling_counts: Dict) -> Optional[Dict]:
        """Process a single record for structural similarity."""
        record_id = record['id']
        predicted_schemas = record['schemas']

        if not predicted_schemas:
            return {
                'id': record_id,
                'schemas': [],
                'label_schemas': [],
                'distances': [],
                'scores': [],
                'triple_scores': [],
                'SS': 0
            }

        xml_data = self.parse_xml_file(record_id)
        if not xml_data:
            return None
        triples, labels = xml_data

        results = []
        total_score = 0
        for schema in predicted_schemas:
            pred_sub = schema['sub']
            pred_rel = schema['rel']
            pred_obj = schema['obj']

            label = next((lbl for lbl in labels if lbl[1] == pred_rel), None)
            if not label:
                results.append({
                    'predicted_schema': schema,
                    'label_schema': None,
                    'distances': {"sub_D": None, "sub_d": None, "obj_D": None, "obj_d": None},
                    'scores': {"sub": 0, "rel": pred_rel, "obj": 0},
                    'triple_score': 0
                })
                continue

            true_sub, _, true_obj = label
            sub_D, sub_d, sub_score = self.calculate_ss_score(hierarchy, child_counts, sibling_counts, true_sub, pred_sub)
            obj_D, obj_d, obj_score = self.calculate_ss_score(hierarchy, child_counts, sibling_counts, true_obj, pred_obj)
            triple_score = sub_score * obj_score
            total_score += triple_score

            results.append({
                'predicted_schema': schema,
                'label_schema': {"sub": true_sub, "rel": pred_rel, "obj": true_obj},
                'distances': {"sub_D": sub_D, "sub_d": sub_d, "obj_D": obj_D, "obj_d": obj_d},
                'scores': {"sub": sub_score, "rel": pred_rel, "obj": obj_score},
                'triple_score': triple_score
            })

        # Calculate SS score with normalization
        if labels and len(labels) >= len(predicted_schemas):
            SS = total_score / len(labels)
        elif labels and len(labels) < len(predicted_schemas):
            SS = (total_score * (len(labels) / len(predicted_schemas)) ** 2) / len(labels)
        else:
            SS = 0

        return {
            'id': record_id,
            'schemas': predicted_schemas,
            'label_schemas': [res['label_schema'] for res in results],
            'distances': [res['distances'] for res in results],
            'scores': [res['scores'] for res in results],
            'triple_scores': [res['triple_score'] for res in results],
            'SS': SS
        }


    def save_results(self, results: List[Dict], filename: str):
        """Save results to JSONL file."""
        with open(filename, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def combine_results(self, triple_metrics: List[Dict], ss_scores: Dict, overall_ss: float):
        """Combine evaluation results."""
        combined = []

        # Group triple metrics by group
        triple_data = {}
        for metric in triple_metrics:
            group = str(metric["group"])
            triple_data[group] = {
                "triple_precision": metric["precision"],
                "triple_recall": metric["recall"],
                "triple_micro_f1": metric["micro_f1"],
                "triple_macro_f1": metric["macro_f1"]
            }

        # Combine with SS scores
        for group in ["1", "2", "3", "Overall"]:
            if group == "1":
                group_name = f"{group} triple"
            elif group in ["2", "3"]:
                group_name = f"{group} triples"
            else:
                group_name = group

            combined.append({
                "group": group_name,
                **triple_data.get(group, {}),
                "SS_score": ss_scores.get(group.split()[0], 0) if group != "Overall" else overall_ss
            })

        self.save_results(combined, self.overall_eval_file)


    def parse_xml_file(self, entry_id: str) -> Optional[Tuple]:
        """Parse XML file for a specific entry ID."""
        xml_file_prefix = '_'.join(entry_id.split('_')[:2])
        xml_file = os.path.join(self.test_folder, f"{xml_file_prefix}.xml")
        if not os.path.exists(xml_file):
            print(f"XML file {xml_file} not found for {entry_id}")
            return None

        tree = ET.parse(xml_file)
        root = tree.getroot()

        for entry in root.findall(".//entry"):
            if entry.get('id') == entry_id:
                triples = []
                schemas = []
                for triple in entry.findall('triples/triple'):
                    sub = triple.find('sub').text.replace('_', ' ').lower()
                    rel = triple.find('rel').text
                    obj = triple.find('obj').text.replace('_', ' ').lower()
                    triples.append((sub, rel, obj))
                for schema in entry.findall('schemas/schema'):
                    sub = schema.find('sub').text
                    rel = schema.find('rel').text
                    obj = schema.find('obj').text
                    schemas.append((sub, rel, obj))
                return triples, schemas
        return None


    def calculate_scores_from_ss(self, ss_data: List[Dict]) -> Tuple[Dict, float]:
        """Calculate group and overall scores from SS data."""
        prefix_scores = defaultdict(list)
        total_scores = []

        for record in ss_data:
            prefix = record['id'].split('_')[0]
            prefix_scores[prefix].append(record['SS'])
            total_scores.append(record['SS'])

        prefix_avg_scores = {
            prefix: round(sum(scores) / len(scores), 3) for prefix, scores in prefix_scores.items()
        }
        overall_avg = round(sum(total_scores) / len(total_scores), 3) if total_scores else 0

        return prefix_avg_scores, overall_avg


    def natural_sort_key(self, s: str):
        """Natural sort key for filenames."""
        def convert(text):
            return int(text) if text.isdigit() else text.lower()
        return [convert(c) for c in re.split('([0-9]+)', s)]