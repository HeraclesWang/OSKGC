import os
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple
from evaluator.base_evaluator import BaseEvaluator


class JointEvaluator(BaseEvaluator):
    def __init__(self, config: Dict):
        super().__init__(config)
        base_name = f"{config['model_name']}_{config['general']['approach']}_{config['general']['example_selection']}"
        self.response_folder = f"LLM_response/{config['model_name']}/{config['general']['approach']}_{config['general']['example_selection']}_{config['model_name']}"
        self.test_folder = config['paths']['test']

        if not os.path.exists(self.response_folder):
            raise FileNotFoundError(f"Response folder not found: {self.response_folder}")

    def evaluate(self):
        """Run full joint approach evaluation."""
        triple_metrics = self.process_triple_evaluation()
        ss_prefix_scores, ss_overall_avg = self.process_structural_similarity()
        self.combine_results(triple_metrics, ss_prefix_scores, ss_overall_avg)

    def process_triple_evaluation(self) -> List[Dict]:
        """Evaluate triple extraction performance for joint approach."""
        all_files_results = []
        group_results = defaultdict(lambda: {"predicted": [], "actual": [], "f1_list": []})
        all_predicted = []
        all_actual = []
        all_f1 = []

        json_files = [f for f in os.listdir(self.response_folder) if f.endswith(".json")]
        json_files = sorted(json_files, key=self.natural_sort_key)

        for json_file in json_files:
            json_path = os.path.join(self.response_folder, json_file)
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
            group_results[group_number]["actual"].extend(
                [triple for record in xml_data.values() for triple in record])
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
        """Parse joint extraction response."""
        pattern = r"\[([^,]+),([^,]+),([^\]]+)\]"
        matches = re.findall(pattern, response)
        return [(sub.strip().lower(), rel.strip().lower(), obj.strip().lower())
                for sub, rel, obj in matches]

    def process_schema_stage(self) -> List[Dict]:
        """Process schema generation stage from response files."""
        output_data = []
        pattern = r"\[.*?\]:\((.*?)\)"

        for file_name in os.listdir(self.response_folder):
            if file_name.endswith(".json"):
                file_path = os.path.join(self.response_folder, file_name)

                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                    for record in data:
                        record_id = record["id"]
                        response = record.get("response", "")
                        schemas = []
                        matches = re.findall(pattern, response)

                        for match in matches:
                            types = match.split(", ")
                            if len(types) == 3:
                                sub_type, rel, obj_type = types
                                schemas.append({
                                    "sub": sub_type.strip(),
                                    "rel": rel.strip(),
                                    "obj": obj_type.strip()
                                })

                        output_data.append({"id": record_id, "schemas": schemas})

        return output_data