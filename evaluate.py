import os
import argparse
from utils.config_loader import load_config, load_api_config
from evaluator.pipeline_evaluator import PipelineEvaluator
from evaluator.joint_evaluator import JointEvaluator
from typing import Dict
from main import update_output_paths


def get_model_name(config: Dict, api_config: Dict = None) -> str:
    """Extract model name for output files."""
    if api_config:
        provider = api_config['api_provider'].lower()
        return api_config[provider]['model_name'].replace('-', '_')
    else:
        model_path = config['models']['entity_recognition']['name'] if config['general']['approach'] == "pipeline" \
            else config['models']['joint_extraction']['name']
        return os.path.basename(model_path.rstrip('/'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to main config file')
    parser.add_argument('--api-config', help='Path to API config file (optional)')
    args = parser.parse_args()

    config = load_config(args.config)
    api_config = load_api_config(args.api_config) if args.api_config else None

    config = update_output_paths(config, api_config)
    config['model_name'] = get_model_name(config, api_config)

    try:
        if config['general']['approach'] == "pipeline":
            evaluator = PipelineEvaluator(config)
        else:
            evaluator = JointEvaluator(config)

        evaluator.evaluate()
        print("Evaluation completed successfully!")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()