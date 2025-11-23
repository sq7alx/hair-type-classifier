import yaml
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_file}")
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()
