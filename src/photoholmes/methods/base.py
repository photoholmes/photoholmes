import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict, TypeVar, Union

import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Module
from typing_extensions import NotRequired

from photoholmes.utils.generic import load_yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


T = TypeVar("T", NDArray, Tensor)


class BenchmarkOutput(TypedDict):
    """
    Structure of the output expected from the benchmark method.

    Expected keys:
        - heatmap: a probability map predicted by the method.
        - mask: a binary mask predicted by the method.
        - detection: score between 0 and 1, where 1 indicates a forged image.

    Extra keys:
        - extra_outputs: any extra outputs that the method might have
            and could be useful to save.
    """

    heatmap: Optional[Tensor]
    mask: Optional[Tensor]
    detection: Optional[Tensor]
    extra_outputs: NotRequired[Dict[str, Any]]


class BaseMethod(ABC):
    """
    Abstract class as a base for the methods.
    Every method should inherit from this class (or BaseTorchMethod) and implement the
    'predict', 'benchmark' and '__init__' methods to enable their utility within the library.
    """

    device: torch.device

    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def predict(self, *args, **kwargs) -> Any:
        """
        Runs method on an image's data, and returns the output of the original implementation.
        """
        raise NotImplementedError("Method `predict` not implemented")

    @abstractmethod
    def benchmark(self, *args, **kwargs) -> BenchmarkOutput:
        """
        Runs method on an image's data and returns the output in the benchmark
        format (BenchMarkOutput).
        """
        return {"heatmap": None, "mask": None, "detection": None}

    @classmethod
    def from_config(cls, config: Optional[str | Dict[str, Any] | Path]):
        """
        Instantiate the model from configuration dictionary or yaml.

        Params:
            config: path to the yaml configuration or a dictionary with
                    the parameters for the model.
        """
        if isinstance(config, (str, Path)):
            config = load_yaml(config)

        if config is None:
            config = {}

        return cls(**config)

    def to_device(self, device: Union[str, torch.device]):
        """Send the model to the device."""
        logging.warning(
            f"Method {self.__class__} isn't a TorchMethod, so it can't be sent "
            "to a device. If the method utilizes a torch model as a feature extractor,"
            "override this method to sent it to the device."
        )
        self.device = torch.device("cpu")


class BaseTorchMethod(BaseMethod, Module):
    """
    Abstract class as a base for methods that are end-to-end Torch modules.
    The child classes must implement the 'predict', 'benchmark' and '__init__' methods
    to enable their utility within the library.
    """

    def __init__(self, *args, **kwargs) -> None:
        Module.__init__(self, *args, **kwargs)
        BaseMethod.__init__(self)

    def load_weights(self, weights: Union[str, Path, dict]):
        """Load weights from a dictionary or a file when given its path."""
        if isinstance(weights, (str, Path)):
            weights_ = torch.load(weights, map_location=self.device)
        else:
            weights_ = weights

        if "state_dict" in weights_.keys():
            weights_ = weights_["state_dict"]

        self.load_state_dict(
            weights_, assign=True
        )  # FIXME: asign limits torch version to >=2.1

    def to_device(self, device: Union[str, torch.device]):
        """Send the model to the device."""
        self.to(device)
        self.device = torch.device(device)
