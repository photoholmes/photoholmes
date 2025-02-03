# FOCAL: Rethinking Image Forgery Detection via Contrastive Learning and Unsupervised Clustering

This is the implementation of the method by Wu et al. that can be found [here](https://arxiv.org/pdf/2308.09307.pdf). Please be aware that the last time we checked (March 9th 2024) this paper was a preprint and may not have been peer-reviewed.

The code contained in this library was derived from [the original implementation](https://github.com/HighwayWu/FOCAL/tree/main), making only minor changes to fit the PhotoHolmes library structure.

This is a deep learning based method, the weights can be found [here](https://drive.google.com/drive/folders/12ayIO9PU4wvqWqniT3KtH8tCvrZ-M-zd?usp=share_link) under the name [Focal_ViT_weights.pth](https://drive.google.com/file/d/1GQMU8FHwi2K3XkkHhe71bt-RQvuA2VQ4/view?usp=share_link) and [Focal_HRnet_weights.pth](https://drive.google.com/file/d/1O_iyg5Tg_iZ5u_yGcU_MhKVH-c6MIpdR/view?usp=share_link). We last checked this information March 9th 2024, please refer to the authors of the original paper if the weights can not be found.

By March 9th 2024 this method could not be run with mps as device because the method uses some torch functions that are not yet implemented for mps.

## Description

FOCAL is based on a simple but very effective paradigm of contrastive learning and unsupervised clustering for the image forgery detection.
Specifically, FOCAL:

1) Utilizes pixel-level contrastive learning to supervise the high-level forensic feature extraction in an image-by-image manner.
2) Employs an on-the-fly unsupervised clustering algorithm (instead of a trained one) to cluster the learned features into forged/pristine categories, further suppressing the cross-image influence from training data.
3) Allows to further boost the detection performance via simple feature-level concatenation without the need of retraining.

## Full overview

The paper, "Rethinking Image Forgery Detection via Contrastive Learning and Unsupervised Clustering," introduces a novel approach named FOCAL (FOrensic ContrAstive cLustering) for image forgery detection.

This method addresses the limitations of traditional pixel classification algorithms by considering the relative nature of forged versus pristine pixels within an image.

FOCAL employs pixel-level contrastive learning to enhance high-level forensic feature extraction and uses an on-the-fly unsupervised clustering algorithm to categorize these features into forged or pristine, improving detection performance without the need for retraining.

Extensive testing across six public datasets shows significant performance improvements over state-of-the-art methods.

The paper highlights the importance of the relative definition of forgery and pristine conditions within images, offering a fresh perspective and setting a new benchmark for future research in image forgery detection.

## Usage

```python
from photoholmes.methods.focal import Focal, focal_preprocessing

# Read an image
from photoholmes.utils.image import read_image
path_to_image = "path_to_image"
image = read_image(path_to_image)

# Assign the image to a dictionary and preprocess the image
image_data = {"image": image}
input = focal_preprocessing(**image_data)

# Declare the method and use the .to_device if you want to run it on cuda instead of cpu
path_to_weights = {"ViT":"path_to_vit_weights","HRNet":"path_to_hrnet_weights"}
method = Focal(
    weights = path_to_weights,
)
method.to_device("cpu")
device = "cpu"
method.to_device(device)

# Use predict to get the final result
output = method.predict(**input)
```

## Citation

``` bibtex
@article{focal,
  title={Rethinking Image Forgery Detection via Contrastive Learning and Unsupervised Clustering},
  author={H. Wu and Y. Chen and J. Zhou},
  year={2023}
}
```
