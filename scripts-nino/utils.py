import argparse
import yaml

def parse_arguments():
    """Parse command line arguments for configuration settings."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model with MAGRPO and different compression levels."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )
    return parser.parse_args()

def load_config(config_file):
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(f"Error in configuration file: {exc}")
        return {}
    except FileNotFoundError:
        print(f"Configuration file not found: {config_file}")
        return {}