from .realistic_tampering import RealisticTamperingDataset


class RealisticTamperingWebPDataset(RealisticTamperingDataset):
    """
    Realistic Tampering Dataset saved in WebP format.

    Directory structure:
    img_dir (realistic-tampering-dataset)
    ├── Canon_60D
    │   ├── ground-truth
    |   |   └── [masks in PNG]
    │   ├── pristine
    |   |   └── [imgs in TIF]
    │   └── tampered-realistic
    |       └── [imgs in TIF]
    ├── Nikon_D90
    │   └── ...[idem above]
    ├── Nikon_D7000
    │   └── ...[idem above]
    ├── Sony_A57
    │   └── ...[idem above]
    └── readme.md
    """

    IMAGE_EXTENSION = ".webp"
