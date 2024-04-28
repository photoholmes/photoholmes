from photoholmes.preprocessing.pipeline import PreProcessingPipeline

focal_preprocessing = PreProcessingPipeline(
    transforms=[],
    inputs=["image"],
    outputs_keys=["image"],
)
