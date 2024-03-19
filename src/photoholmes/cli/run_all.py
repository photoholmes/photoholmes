from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

from photoholmes.methods.registry import MethodRegistry
from photoholmes.utils.generic import load_yaml
from photoholmes.utils.image import read_image, read_jpeg_data


class MethodConfig(BaseModel):
    method: MethodRegistry
    config: Dict[str, Any]


class RunAllConfig(BaseModel):
    images: List[Path]
    methods: List[MethodConfig]


def load_image(image_path: Path, inputs: List[str]):
    inp = {}

    if "image" in inputs:
        inp["image"] = read_image(str(image_path))

    if "dct_coefficients" in inputs or "qtables" in inputs:
        dct_coefs, qtables = read_jpeg_data(str(image_path))

        if "dct_coefficients" in inputs:
            inp["dct_coefficients"] = dct_coefs
        if "qtables" in inputs:
            inp["qtables"] = qtables

    return inp


def run_all_methods(config_path: Path):
    from photoholmes.methods.factory import MethodFactory

    config = RunAllConfig(**load_yaml(config_path))

    for m in config.methods:
        method, preprocessing = MethodFactory.load(m.method, m.config)

        for i in config.images:
            image_data = load_image(image_path, preprocessing.inputs)

    pass
