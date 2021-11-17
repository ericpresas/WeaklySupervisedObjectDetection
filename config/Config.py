import yaml


def build_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


config = build_config('config/coco.yaml')