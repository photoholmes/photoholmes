# EXIF as Language: Learning Cross-Modal Associations Between Images and Camera Metadata

This is the implemetation fo the Exif as Language method by Zheng et al. that can be found [here](https://arxiv.org/pdf/2301.04647.pdf).

The code contained in this library was derived from [the original implementation](https://github.com/hellomuffin/exif-as-language), making only minor changes to fit the PhotoHolmes library structure.

This is a deep learning based method, the weights can be found [here](https://drive.google.com/drive/folders/1V9g3I2SoQtjAUz71hZeMutqoGpUiPl3u) under the name [wrapper_75_new.pth](https://drive.google.com/file/d/17MW-fZRRQQ8dSRv52X_9DmcmdQD7TmHZ/view?usp=share_link). We last checked this information on March 9th 2024, please refer to the authors of the original paper if the weights cannot be found.

Please be aware that in order to use the provided weights with PhotoHolmes you should run the PhotoHolmes CLI with the `adapt-weights` command to prune the weights or use the file `prune_original_weights.py`

## Description

An image file contains not only the pixel values, but also a lot of extra-metadata that accompanies the image taken: camera model, exposure time, focal length, jpeg quantization details, etc. In this method the content of the image is contrasted with the exif information to detect any inconsistencies between what is "said" about the image and what the image is.

## Full overview

The method consist of training both an image and text encoder through contrastive learning, obtaining a single, cross-modal embedding space. The paper draws inspiration from openai's CLIP, hanging out the natural language for the EXIF information concatenated as a string.

The result of this training scheme are two encoder, one image and text, that work in the same embedding space. In other words, patches from the same image should be close in the embedding space, while patches from  images that have different EXIF information shouldn't be close. This allows us to use the image embedder to detect images that have been spliced. If patches taken from the same image cluster in two or more regions of the embedding space, that means that the image is a splicing of images that share different EXIF data

## Usage

The following example assumes you have alread prunned the weights as explained before.

```python
from photoholmes.methods.exif_as_language import (
    EXIFAsLanguage,
    exif_as_language_preprocessing,
)

# Read an image
from photoholmes.utils.image import read_image
path_to_image = "path_to_image"
image = read_image(path_to_image)

# Assign the image to a dictionary and preprocess the image
image_data = {"image": image}
input = exif_as_language_preprocessing(**image_data)

# Declare the method and use the .to_device if you want to run it on cuda or mps instead of cpu
arch_config = "pretrained"
path_to_weights = "path_to_weights"
method = EXIFAsLanguage(
    arch_config = arch_config,
    weights = path_to_weights,
)
device = "cpu"
method.to_device(device)

# Use predict to get the final result
output = method.predict(**input)
```

## Citation

``` bibtex
@article{zheng2023exif,
  title={EXIF as Language: Learning Cross-Modal Associations Between Images and Camera Metadata},
  author={Zheng, Chenhao and Shrivastava, Ayush and Owens, Andrew},
  journal={arXiv preprint arXiv:2301.04647},
  year={2023}
}
```
