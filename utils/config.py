import yaml
import argparse

# def load_config(path):
#     with open(path, 'r') as f:
#         return yaml.safe_load(f)

# from omegaconf import OmegaConf

# def load_config(path: str):
#     return OmegaConf.load(path)

from omegaconf import OmegaConf
import os

from omegaconf import OmegaConf
import os

def load_config(config_path: str):
    """
    Load a config file with support for `defaults:`.
    If defaults are present, load them from the same directory as config_path.
    """
    cfg = OmegaConf.load(config_path)

    # If no defaults, return as-is
    if "defaults" not in cfg:
        return cfg

    # Get the directory where config lives
    config_dir = os.path.dirname(config_path)
    defaults = cfg.defaults
    del cfg["defaults"]

    # Merge all defaults (assumed to be filenames in same folder)
    merged_cfg = OmegaConf.create()
    for default_file in defaults:
        if not isinstance(default_file, str):
            raise ValueError(f"Only string-based defaults are supported, got: {default_file}")
        
        default_path = os.path.join(config_dir, f"{default_file}.yml")
        if not os.path.exists(default_path):
            raise FileNotFoundError(f"Default config '{default_path}' not found")
        
        sub_cfg = OmegaConf.load(default_path)
        merged_cfg = OmegaConf.merge(merged_cfg, sub_cfg)

    # Merge the current config over the defaults
    return OmegaConf.merge(merged_cfg, cfg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    return parser.parse_args()
