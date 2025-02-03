# DQ: Fast, Automatic, and Fine-Grained Tampered JPEG Image Detection via DCT Coefficient Analysis

This is the implementation of the method by Lin et al. that can be found [here](http://mmlab.ie.cuhk.edu.hk/archive/2009/pr09_fast_automatic.pdf).

## Description

With the rapid advancement in image/video editing techniques, distinguishing tampered images from real ones has become a challenge. This method focuses on JPEG images and detects tampered regions by examining the double quantization effect hidden among the discrete cosine transform (DCT) coefficients.

The method offers:

    Automatic Location: It can automatically locate the tampered region without user intervention.
    Fine-Grained Detection: The detection is at the scale of 8x8 DCT blocks.
    Versatility: Capable of dealing with images tampered using various methods such as inpainting, alpha matting, texture synthesis, and other editing skills.
    Efficiency: Directly analyzes the DCT coefficients without fully decompressing the JPEG image, ensuring fast performance.

## Full Overview

The method is based on the DQ effect in forged JPEG images and can produce fine-grained output of the forgery region at the scale of 8x8 image blocks. The algorithm directly analyzes the DCT coefficients without fully decompressing the JPEG image, saving memory and computational load. The method is faster than bi-coherence based approaches and CRF based algorithms.

## Usage

```python
from photoholmes.methods.dq import DQ, dq_preprocessing

# Read an image
from photoholmes.utils.image import read_image, read_jpeg_data
path_to_image = "path_to_image"
image = read_image(path_to_image)
dct, qtables = read_jpeg_data(path_to_image)

# Assign the image, dct and qtables to a dictionary and preprocess the image
image_data = {"image": image, "dct_coefficients": dct}
input = dq_preprocessing(**image_data)

# Declare the method
method = DQ()

# Use predict to get the final result
output = method.predict(**input)   
```

## Citation

``` bibtex
@ARTICLE{FastJPEGDetection2009,
  author={Zhouchen Lin, Junfeng He, Xiaoou Tang, Chi-Keung Tang},
  journal={Pattern Recognition},
  title={Fast, automatic and fine-grained tampered JPEG image detection via DCT coefficient analysis},
  year={2009},
  doi={10.1016/j.patcog.2009.03.019}
}
```
