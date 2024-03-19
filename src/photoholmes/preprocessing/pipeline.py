import logging
from typing import Any, Dict, List, Literal

from photoholmes.preprocessing.base import BasePreprocessing

logger = logging.getLogger(__name__)


class PreProcessingPipeline:
    """
    A pipeline of preprocessing transforms. In this library, the standard way of defining
    the preprocessing of a method is by creating an instance of this class with the corresponding
    sequence of transforms.
    """

    inputs: List[Literal["image", "dct_coefficients", "qtables"]]
    outputs_keys: List[str]

    def __init__(
        self,
        transforms: List[BasePreprocessing],
        inputs: List[Literal["image", "dct_coefficients", "qtables"]],
        outputs_keys: List[str],
    ) -> None:
        """
        Initializes a new preprocessing pipeline.

        Args:
            transforms (List[BasePreprocessing]): A list of preprocessing transforms to
                apply to the input.
            inputs (List[str]): the inputs that the pipeline will receive.
            outputs_keys (List[str]): the keys of the outputs that the pipeline will
                return. These must coincide with the keyword arguments of the predict and benchmark methods.
        """
        self.transforms = transforms
        self.inputs = inputs
        self.outputs_keys = outputs_keys

    def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Applies the preprocessing pipeline to the input.

        Args:
            **kwargs: Keyword arguments representing the input to the pipeline.

        Returns:
            Dict[str, Any]: A dictionary with the output of the last transform in the
                            pipeline.
        """
        self._check_inputs(kwargs)

        for t in self.transforms:
            kwargs = t(**kwargs)

        return {k: v for k, v in kwargs.items() if k in self.outputs_keys}

    def _check_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Checks the inputs required are included in the ones declared in the pipeline and raises
         a Warning for any input not used in the pipeline.
        """
        for input_ in self.inputs:
            if input_ not in inputs:
                raise ValueError(f"Missing input {input_} in inputs")

        for input_ in inputs.keys():
            if input_ not in self.inputs:
                logger.warn(f"Input {input_} is not used by the pipeline")

    def append(self, transform: BasePreprocessing):
        self.transforms.append(transform)

    def insert(self, index: int, transform: BasePreprocessing):
        self.transforms.insert(index, transform)
