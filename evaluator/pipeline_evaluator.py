import os
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple
from evaluator.base_evaluator import BaseEvaluator


class PipelineEvaluator(BaseEvaluator):
    def __init__(self, config: Dict):
        super().__init__(config)
        base_name = f"{config['model_name']}_{config['general']['approach']}_{config['general']['example_selection']}"
        self.entity_typing_folder = f"LLM_response/{config['model_name']}/{config['general']['approach']}_{config['general']['example_selection']}/Typing_step_{base_name}"
        self.relation_selection_folder = f"LLM_response/{config['model_name']}/{config['general']['approach']}_{config['general']['example_selection']}/Relation_step_{base_name}"
        self.test_folder = config['paths']['test']

        if not os.path.exists(self.entity_typing_folder):
            raise FileNotFoundError(f"Entity typing folder not found: {self.entity_typing_folder}")
        if not os.path.exists(self.relation_selection_folder):
            raise FileNotFoundError(f"Relation selection folder not found: {self.relation_selection_folder}")

    def evaluate(self):
        """Run full pipeline evaluation."""
        triple_metrics = self.process_triple_evaluation()
        ss_prefix_scores, ss_overall_avg = self.process_structural_similarity()
        self.combine_results(triple_metrics, ss_prefix_scores, ss_overall_avg)

    def process_triple_evaluation(self) -> List[Dict]:
        """Evaluate triple extraction performance for pipeline approach."""
        all_files_results = []
        group_results = defaultdict(lambda: {"predicted": [], "actual": [], "f1_list": []})
        all_predicted = []
        all_actual = []
        all_f1 = []

        json_files = [f for f in os.listdir(self.relation_selection_folder) if f.endswith(".json")]
        json_files = sorted(json_files, key=self.natural_sort_key)

        for json_file in json_files:
            json_path = os.path.join(self.relation_selection_folder, json_file)
            xml_path = os.path.join(self.test_folder, os.path.splitext(json_file)[0] + ".xml")

            if not os.path.exists(xml_path):
                print(f"Skipping {json_file} as corresponding XML file is missing.")
                continue

            xml_data = self.parse_xml(xml_path)
            results, overall_precision, overall_recall, overall_f1, macro_f1 = self.process_file(json_path, xml_data)

            # Save detailed results per file
            output_jsonl_path = os.path.join(self.detailed_dir, os.path.splitext(json_file)[0] + ".jsonl")
            self.save_results(results, output_jsonl_path)

            all_files_results.append({
                "file_name": json_file,
                "precision": overall_precision,
                "recall": overall_recall,
                "micro_f1": overall_f1,
                "macro_f1": macro_f1
            })

            group_number = int(json_file.split("_")[0])
            group_results[group_number]["predicted"].extend(
                [triple for record in results for triple in self.parse_response(record["response"])])
            group_results[group_number]["actual"].extend([triple for record in xml_data.values() for triple in record])
            group_results[group_number]["f1_list"].append(macro_f1)

            all_predicted.extend([triple for record in results for triple in self.parse_response(record["response"])])
            all_actual.extend([triple for record in xml_data.values() for triple in record])
            all_f1.append(macro_f1)

        # Calculate overall metrics
        overall_precision, overall_recall, overall_micro_f1 = self.calculate_metrics(all_predicted, all_actual)
        overall_macro_f1 = round(sum(all_f1) / len(all_f1), 3) if all_f1 else 0

        # Prepare group metrics
        group_metrics = []
        for group, data in sorted(group_results.items()):
            precision, recall, micro_f1 = self.calculate_metrics(data["predicted"], data["actual"])
            macro_f1 = round(sum(data["f1_list"]) / len(data["f1_list"]), 3) if data["f1_list"] else 0
            group_metrics.append({
                "group": group,
                "precision": precision,
                "recall": recall,
                "micro_f1": micro_f1,
                "macro_f1": macro_f1
            })

        # Add overall metrics
        group_metrics.append({
            "group": "Overall",
            "precision": overall_precision,
            "recall": overall_recall,
            "micro_f1": overall_micro_f1,
            "macro_f1": overall_macro_f1
        })

        # Save triple evaluation results
        self.save_results(all_files_results, self.triple_eval_file)

        return group_metrics


    def parse_response(self, response: str) -> List[Tuple]:
        """Parse relation selection response."""
        pattern = r"\[([^\]]+)\]\{([^,]+),\s*([^}]+)\}"
        matches = re.findall(pattern, response)
        return [(sub.strip().lower(), rel.strip().lower(), obj.strip().lower())
                for rel, sub, obj in matches]

    def process_schema_stage(self) -> List[Dict]:
        """Process first stage to generate schema data."""
        results = []

        for file_et in os.listdir(self.entity_typing_folder):
            if not file_et.endswith(".json"):
                continue

            with open(os.path.join(self.entity_typing_folder, file_et), "r", encoding='utf-8') as f_et:
                data_et = json.load(f_et)

            file_rs = file_et
            if not os.path.exists(os.path.join(self.relation_selection_folder, file_rs)):
                continue

            with open(os.path.join(self.relation_selection_folder, file_rs), "r", encoding='utf-8') as f_rs:
                data_rs = json.load(f_rs)

            for entry_et, entry_rs in zip(data_et, data_rs):
                if entry_et["id"] != entry_rs["id"]:
                    continue

                entities = self.parse_entities(entry_et.get("response", ""))
                relations = self.parse_relations(entry_rs.get("response", ""))

                schemas = []
                for relation, head_entity, tail_entity in relations:
                    head_type = entities.get(head_entity, None)
                    tail_type = entities.get(tail_entity, None)
                    if head_type and tail_type:
                        schemas.append({
                            "sub": head_type,
                            "rel": relation,
                            "obj": tail_type
                        })

                results.append({
                    "id": entry_et["id"],
                    "schemas": schemas
                })

        return results

    def parse_entities(self, response: str) -> Dict[str, str]:
        """Extract entities and their types from entity typing response."""
        entities = {}
        matches = re.findall(r"\{([^}]+)\}:\[([^\]]+)\]", response)
        for entity, entity_type in matches:
            processed_type = entity_type.strip()
            if ':' in processed_type:
                processed_type = processed_type.split(':')[-1].strip()
            processed_type = re.sub(r'[\[\]\{\}]', '', processed_type).strip()
            entities[entity.strip()] = processed_type
        return entities

    def parse_relations(self, response: str) -> List[Tuple]:
        """Extract relations from relation selection response."""
        relations = []
        matches = re.findall(r"\[([^\]]+)\]\{([^,]+),\s*([^}]+)\}", response)
        for relation, head, tail in matches:
            relations.append((relation.strip(), head.strip(), tail.strip()))
        return relations