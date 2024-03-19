from photoholmes.preprocessing.image import ZeroOneRange
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

trufor_preprocessing = PreProcessingPipeline(
    inputs=["image"], outputs_keys=["image"], transforms=[ZeroOneRange()]
)
