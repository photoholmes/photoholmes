from io import BytesIO
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor

from photoholmes.preprocessing import (
    BasePreprocessing,
    PreProcessingPipeline,
    RGBtoGray,
    RoundToUInt,
    ToNumpy,
)


class AddImage99(BasePreprocessing):
    def __call__(self, image: Tensor | NDArray, **kwargs) -> Dict[str, Any]:

        if isinstance(image, Tensor):
            image = image.permute(1, 2, 0).numpy()

        img_pil = Image.fromarray(image)

        f = BytesIO()
        img_pil.save(f, "JPEG", quality=99, subsampling=0)
        f.seek(0)
        image_99 = Image.open(f)
        image_99 = np.array(image_99)
        f.close()

        return {"image": image, "image_99": image_99, **kwargs}


zero_preprocessing = PreProcessingPipeline(
    inputs=["image"],
    outputs_keys=["image", "image_99"],
    transforms=[
        ToNumpy(),
        AddImage99(),
        RGBtoGray(extra_image_keys=["image_99"]),
        RoundToUInt(apply_on=["image", "image_99"]),
    ],
)
