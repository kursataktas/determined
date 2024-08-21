"""
Script for retagging docker images based on `old` and `new` keys found
in bumpenvs.yaml.
This script is intended to be used by release party to retag hashed
images as rc-0,rc-1, and release version.
This script assumes that the `old` image tag exists, but the `new` tag
does not. It will use the `old`

Usage:
    python update-docker-tags.sh path/to/bumpenvs.yaml [--target_version]

Args:
    path: str
        Path to bumpenvs.yaml file.
Optional Args
    target_version: str
        Version in `old` to retag.
"""


import argparse
import pathlib

from python_on_whales import docker

try:
    from ruamel import yaml
except ModuleNotFoundError:
    # Inexplicably, sometimes ruamel.yaml is packaged as ruamel_yaml instead.
    import ruamel_yaml as yaml  # type: ignore

def gather_images(yaml_path: str, target_environment: str) -> list:
    with open(yaml_path) as f:
        images = yaml.YAML(typ="safe", pure=True).load(f)
    valid_images = []

    for image in images.values():
        if target_environment in image["new"]:
            valid_images.append(image)

    return valid_images


def main(yaml_path: str, target_version: str) -> None:
    images = gather_images(yaml_path, target_version)
    for image in images:
        docker.buildx.imagetools.create(sources=[image["old"]], tags=[image["new"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=pathlib.Path, help="path/to/bumpenvs.yaml")
    parser.add_argument(
        "--target_environment", help="version id (default found in environments-target.txt)"
    )
    args = parser.parse_args()

    if args.target_environment is None:
        with open("tools/scripts/environments-target.txt") as f:
            target_environment = f.readline()
    else:
        target_environment = args.target_environment

    main(args.path, target_environment)
