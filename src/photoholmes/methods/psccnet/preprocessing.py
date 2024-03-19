from photoholmes.preprocessing.image import ZeroOneRange
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

psccnet_preprocessing = PreProcessingPipeline(
    inputs=["image"],
    outputs_keys=["image"],
    transforms=[
        ZeroOneRange(),
    ],
)
