from photoholmes.preprocessing.image import ToNumpy
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

noisesniffer_preprocessing = PreProcessingPipeline(
    inputs=["image"],
    outputs_keys=["image"],
    transforms=[ToNumpy()],
)
