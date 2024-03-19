from photoholmes.preprocessing.image import ZeroOneRange
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

focal_preprocessing = PreProcessingPipeline(
    transforms=[ZeroOneRange()],
    inputs=["image"],
    outputs_keys=["image"],
)
