import yaml
import os


def load_config(config_path: str) -> dict:
    """Load main configuration file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Validate paths
    required_paths = ['train', 'test', 'ontology', 'hierarchy_xml']
    for path in required_paths:
        if not os.path.exists(config['paths'][path]):
            raise FileNotFoundError(f"Required path not found: {config['paths'][path]}")

    if config['general']['example_selection'] == "example" and not os.path.exists(config['paths']['example']):
        raise FileNotFoundError(f"Example path not found: {config['paths']['example']}")

    return config


def load_api_config(api_config_path: str) -> dict:
    """Load API configuration file"""
    with open(api_config_path, 'r') as file:
        api_config = yaml.safe_load(file)

    # Validate API config
    provider = api_config.get('api_provider', '').lower()
    if provider not in ['gemini', 'claude', 'openai']:
        raise ValueError("Invalid API provider in config_api.yaml")

    if not api_config.get(provider, {}).get('api_key'):
        raise ValueError(f"Missing API key for {provider}")

    # Clean model name for path usage
    api_config[provider]['model_name_clean'] = api_config[provider]['model_name'].replace('-', '_')

    return api_config