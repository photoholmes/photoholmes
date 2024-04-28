from .columbia import ColumbiaDataset


class ColumbiaWebPDataset(ColumbiaDataset):
    """
    Class for the Columbia Uncompressed Image Splicing Detection dataset, saved in webp format.

    Directory structure:
    img_dir (Columbia Uncompressed Image Splicing Detection)
    ├── 4cam_auth
    │   ├── [images in WEBP]
    └── 4cam_splc
        ├── [images in WEBP]
        └── edgemask
            └── [masks in JPG]
    """

    IMAGE_EXTENSION = ".webp"
