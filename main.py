import yaml
import argparse
from utils.config_loader import load_config, load_api_config
from pipeline.entity_recognition import run_entity_recognition
from pipeline.entity_typing import run_entity_typing
from pipeline.relation_selection import run_relation_selection
from joint.extraction import run_joint_extraction
import os
from evaluator.pipeline_evaluator import PipelineEvaluator
from evaluator.joint_evaluator import JointEvaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to main config file')
    parser.add_argument('--api-config', help='Path to API config file (optional)')
    parser.add_argument('--do', choices=['run', 'evaluate', 'both'], default='run',
                      help='What to do: run model, evaluate results, or both')
    return parser.parse_args()


def get_model_name(config, api_config=None):
    """Get clean model name for path usage"""
    if api_config:
        provider = api_config['api_provider'].lower()
        return api_config[provider]['model_name'].replace('-', '_')
    else:
        approach = config['general']['approach']
        model_path = config['models']['entity_recognition']['name'] if approach == "pipeline" else \
        config['models']['joint_extraction']['name']
        return os.path.basename(model_path.rstrip('/'))


def update_output_paths(config, api_config=None):
    """Update output paths based on model names and settings"""
    approach = config['general']['approach']
    example_selection = config['general']['example_selection']
    model_name = get_model_name(config, api_config)

    # Base output directory naming
    base_name = f"{model_name}_{approach}_{example_selection}"

    if 'output' not in config:
        config['output'] = {}

    if approach == "pipeline":
        if 'pipeline' not in config['output']:
            config['output']['pipeline'] = {}
        config['output']['pipeline']['entity_recognition'] = f"LLM_response/{model_name}/{approach}_{example_selection}/NER_step_{base_name}"
        config['output']['pipeline']['entity_typing'] = f"LLM_response/{model_name}/{approach}_{example_selection}/Typing_step_{base_name}"
        config['output']['pipeline']['relation_selection'] = f"LLM_response/{model_name}/{approach}_{example_selection}/Relation_step_{base_name}"
    else:
        config['output']['joint'] = f"LLM_response/{model_name}/{approach}_{example_selection}_{model_name}"

    return config


def main():
    args = parse_args()

    # Load configurations
    config = load_config(args.config)
    api_config = load_api_config(args.api_config) if args.api_config else None

    config = update_output_paths(config, api_config)
    config['model_name'] = get_model_name(config, api_config)

    if args.do in ['run', 'both']:
        if config['general']['approach'] == "pipeline":
            print("Running pipeline approach...")

            # Step 1: Entity Recognition
            print("\nRunning Entity Recognition...")
            er_output = run_entity_recognition(config, api_config)

            # Step 2: Entity Typing
            print("\nRunning Entity Typing...")
            et_output = run_entity_typing(config, api_config, er_output)

            # Step 3: Relation Selection
            print("\nRunning Relation Selection...")
            rs_output = run_relation_selection(config, api_config, et_output)

            print("\nPipeline execution completed!")
        else:
            print("Running joint approach...")
            run_joint_extraction(config, api_config)
            print("\nJoint extraction completed!")

    if args.do in ['evaluate', 'both']:
        if config['general']['approach'] == "pipeline":
            evaluator = PipelineEvaluator(config)
        else:
            evaluator = JointEvaluator(config)

        evaluator.evaluate()
        print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()