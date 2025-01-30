# CAT-Net: Compression Artifact Tracing Network for Detection and Localization of Image Splicing

This is the implemenattion of the CAT-Net method by Kwon et al. that can be found [here](https://openaccess.thecvf.com/content/WACV2021/papers/Kwon_CAT-Net_Compression_Artifact_Tracing_Network_for_Detection_and_Localization_of_WACV_2021_paper.pdf) on its first version and [here](https://arxiv.org/pdf/2108.12947.pdf) on its second version.
Both papers introduce the same architechture, differing only in the trainig dataset used: v1 targeted only splicing while v2 targets splicing and copy-move.

The code contained in this library was derived from [the original implementation](https://github.com/mjkwon2021/CAT-Net), making only minor changes to fit the PhotoHolmes library structure.

This is a deep learning based method, the weights can be found [here](https://drive.google.com/drive/folders/14uNqj46505MQc3swBQgbaiPVAWtNChbz) under the name [CAT_full_v1.pth](https://drive.google.com/file/d/1NXLDCn0ABG7eWEXltGZ4SyIsREhOUhRM/view?usp=share_link) and [CAT_full_v2.pth](https://drive.google.com/file/d/1tyOKVdx6UMys2OcNpUj9r6scxNIpcoLE/view?usp=share_link) We last checked this information on March 9th 2024, please refer to the authors of the original paper if the weights cannot be found.

## Description

CAT-Net is an end-to-end fully convolutional neural network designed to detect compression artifacts in images. CAT-Net combines both RGB and DCT streams, allowing it to simultaneously learn forensic features related to compression artifacts in these domains. Each stream considers multiple resolutions to deal with the various shapes and sizes of the spliced objects.

## Full overview

The RGB stream processes the color information of the image, which is often altered during image splicing, while the DCT stream analyzes the compression artifacts, which are usually introduced when an image is saved in a compressed format like JPEG. By analyzing both color information and compression artifacts, CAT-Net can better discern inconsistencies associated with splicing, compared to using only one of the streams.

Multiple resolution analysis refers to the processing of image data at various scales or resolutions. This is crucial in image splicing detection because spliced objects can come in different sizes and shapes. By analyzing the image at multiple resolutions, CAT-Net can adapt to various scales of splicing, making it more versatile and accurate in detecting spliced regions, irrespective of the size of the spliced objects.

## Usage

```python
from photoholmes.methods.catnet import CatNet, catnet_preprocessing

# Read an image
from photoholmes.utils.image import read_image, read_jpeg_data
path_to_image = "path_to_image"
image = read_image(path_to_image)
dct, qtables = read_jpeg_data(path_to_image)

# Assign the image, dct and qtables to a dictionary and preprocess the image
image_data = {"image": image, "dct_coefficients": dct, "qtables": qtables}
input = catnet_preprocessing(**image_data)

# Declare the method and use the .to_device if you want to run it on cuda or mps instead of cpu
arch_config = "pretrained"
path_to_weights = "path_to_weights"
method = CatNet(
  arch_config=arch_config,
  weights=path_to_weights,
)
device = "cpu"
method.to_device(device)

# Use predict to get the final result
output = method.predict(**input)
```

## Citation

``` bibtex
@inproceedings{kwon2021cat,
  title={CAT-Net: Compression Artifact Tracing Network for Detection and Localization of Image Splicing},
  author={Kwon, Myung-Joon and Yu, In-Jae and Nam, Seung-Hun and Lee, Heung-Kyu},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={375--384},
  year={2021}
}
```

``` bibtex
@article{kwon2022learning,
  title={Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization},
  author={Kwon, Myung-Joon and Nam, Seung-Hun and Yu, In-Jae and Lee, Heung-Kyu and Kim, Changick},
  journal={International Journal of Computer Vision},
  volume = {130},
  number = {8},
  pages={1875--1895},
  month = aug,
  year={2022},
  publisher={Springer},
  doi = {10.1007/s11263-022-01617-5}
}
```
