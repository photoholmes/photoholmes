from photoholmes.preprocessing.image import RGBtoGray, RoundToUInt, ToNumpy
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

zero_preprocessing = PreProcessingPipeline(
    inputs=["image"],
    outputs_keys=["image"],
    transforms=[RGBtoGray(), RoundToUInt(), ToNumpy()],
)
