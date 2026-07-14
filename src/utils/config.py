import os
import yaml


def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def merge_cli_overrides(config, overrides):
    for key, value in overrides.items():
        if value is None:
            continue
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return config
