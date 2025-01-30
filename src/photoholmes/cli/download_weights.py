import logging
import os
from functools import partial
from pathlib import Path

import wget

logger = logging.getLogger(__name__)


def callback(current, total, width=80, message: str = "Downloading..."):
    def format_bytes(num):
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if num < 1024.0:
                return f"{num:.2f} {unit}"
            num /= 1024.0

    def progress_bar(current, total, width=80):
        progress = int(width * current / total)
        bar = "[" + "=" * progress + " " * (width - progress) + "]"
        return bar

    percent = current / total * 100
    progress = progress_bar(current, total, width)
    print(
        f"\r{message}: {percent:.2f}% {progress} {format_bytes(total)}",
        end="",
        flush=True,
    )


def download_psccnet_weights(weights_folder: Path):
    os.makedirs(weights_folder, exist_ok=True)
    wget.download(
        "https://github.com/proteus1991/PSCC-Net/raw/main/checkpoint/HRNet_checkpoint/HRNet.pth",  # noqa: E501
        out=str(weights_folder / "FENet.pth"),
        bar=partial(callback, message="Downloading FENet"),
    )
    print()
    wget.download(
        "https://github.com/proteus1991/PSCC-Net/raw/main/checkpoint/NLCDetection_checkpoint/NLCDetection.pth",  # noqa: E501
        out=str(weights_folder / "SegNet.pth"),
        bar=partial(callback, message="Downloading SegNet"),
    )
    print()
    wget.download(
        "https://github.com/proteus1991/PSCC-Net/raw/main/checkpoint/DetectionHead_checkpoint/DetectionHead.pth",  # noqa: E501
        out=str(weights_folder / "ClsNet.pth"),
        bar=partial(callback, message="Downloading ClsNet"),
    )
    print()
    logger.info(f"Downloaded PSCC-Net weights to {weights_folder}")


def download_exif_weights(weights_folder: Path):
    os.makedirs(weights_folder, exist_ok=True)
    wget.download(
        "https://drive.usercontent.google.com/download?id=1EdEzBhCY9gO3qYvoNMibk8SwAxerDtQp&export=download&authuser=0&confirm=t&uuid=d7fc01e8-e819-4d54-85fc-6013674c1397&at=AIrpjvP0YWIlV0pcXULfNBk_7TUv%3A1738204986471",  # noqa: 501
        out=str(weights_folder / "weights.pth"),
        bar=partial(callback, message="Downloading weights"),
    )
    print()
    logger.info(f"Downloaded Exif as Language weights to {weights_folder}")


def download_adaptive_cfa_net_weights(weights_folder: Path):
    os.makedirs(weights_folder, exist_ok=True)
    wget.download(
        "https://raw.githubusercontent.com/qbammey/adaptive_cfa_forensics/master/src/models/pretrained.pt",  # noqa: E501
        out=str(weights_folder / "weights.pth"),
        bar=partial(callback, message="Downloading weights"),
    )
    print()
    logger.info(f"Downloaded Adaptive CFA Net weights to {weights_folder}")


def download_catnet_weights(weights_folder: Path):
    os.makedirs(weights_folder, exist_ok=True)
    wget.download(
        "https://drive.usercontent.google.com/download?id=1tyOKVdx6UMys2OcNpUj9r6scxNIpcoLE&export=download&authuser=0&confirm=t&uuid=bfd131d7-972a-4679-af6e-893d6eff70ca&at=APZUnTVFGzcrFlrKeCuuVrAqRgwv:1710203713701",  # noqa: E501
        out=str(weights_folder / "weights.pth"),
        bar=partial(callback, message="Downloading weights"),
    )
    print()
    logger.info(f"Downloaded CATNet weights to {weights_folder}")


def download_trufor_weights(weights_folder: Path):
    import shutil
    import zipfile

    os.makedirs(weights_folder, exist_ok=True)
    wget.download(
        "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip",
        out=str(weights_folder / "weights.zip"),
        bar=partial(callback, message="Downloading weights"),
    )
    print()
    logger.info(f"Downloaded TruFor weights to {weights_folder}")
    logger.info("Unzipping weights...")
    with zipfile.ZipFile(weights_folder / "weights.zip", "r") as zip_ref:
        zip_ref.extractall(weights_folder)
    shutil.move(
        weights_folder / "weights/trufor.pth.tar", weights_folder / "trufor.pth.tar"
    )

    os.remove(weights_folder / "weights.zip")
    os.rmdir(weights_folder / "weights")


def download_focal_weights(weights_folder: Path):
    logger.info(
        "We are downloading the pruned version of the Focal weights "
        "created by the photoholmes team, removing unused layers. "
        "If you want the original weights, see https://github.com/HighwayWu/FOCAL/tree/main"  # noqa: E501
    )
    os.makedirs(weights_folder, exist_ok=True)
    wget.download(
        "https://drive.usercontent.google.com/download?id=15bkxN0jFUnmVL8vBMbhYdlSFsO8jdebe&export=download&authuser=0&confirm=t&uuid=a17eb15f-e5c0-448c-a620-8816a68bb4d9&at=APZUnTUFUP0rjAN_ukOlUDdYc6Oc:1710209216092",  # noqa: E501
        out=str(weights_folder / "VIT_weights.pth"),
        bar=partial(callback, message="Downloading VIT weights"),
    )
    print()
    wget.download(
        "https://drive.usercontent.google.com/download?id=1FTeFqF0kjjfoB7U6FLxfKER45oeYOoIP&export=download&authuser=0&confirm=t&uuid=08d12cc9-dfd2-4d86-8ada-e109ed15cc90&at=APZUnTU3FCfmnZiWIGC5ZrFGaneU:1710209320576",  # noqa: E501
        out=str(weights_folder / "HRNET_weights.pth"),
        bar=partial(callback, message="Downloading HRNet weights"),
    )
    print()
    logger.info(f"Downloaded Focal weights to {weights_folder}")
