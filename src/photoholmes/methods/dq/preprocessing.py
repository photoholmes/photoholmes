from photoholmes.preprocessing.image import GetImageSize, ToNumpy
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

dq_preprocessing = PreProcessingPipeline(
    inputs=["image", "dct_coefficients"],
    outputs_keys=["dct_coefficients", "image_size"],
    transforms=[
        GetImageSize(),
        ToNumpy(),
    ],
)
