from photoholmes.preprocessing.image import RGBtoGray, ToNumpy, ZeroOneRange
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

splicebuster_preprocessing = PreProcessingPipeline(
    inputs=["image"],
    outputs_keys=["image"],
    transforms=[ZeroOneRange(), RGBtoGray(), ToNumpy()],
)
