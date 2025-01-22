import logging
import os
from math import ceil
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import typer
from matplotlib import pyplot as plt

from photoholmes.utils.image import overlay_mask, read_image, read_jpeg_data

logger = logging.getLogger("cli.run_method")


run_app = typer.Typer(name="run", help="Run a method on an image.")


IMAGE_HELP = "Path to image to analyze."
OUTPUT_FOLDER_HELP = (
    "Path to folder to save outputs. If no path is provided outputs aren't saved."
)
OVERLAY_HELP = "Include mask overlay on the image when plotting the results."
SHOW_PLOT_HELP = "Show results as a matplotlib plot."
DEVICE_HELP = "Select device: 'cuda', 'cpu' or 'mps'"


def plot_results(
    title: str, plots: List[Dict[str, Any]], score: Optional[float] = None
):
    plt.clf()
    plt.suptitle(title)
    row = ceil(len(plots) / 2)
    cols = 2 if len(plots) > 2 else len(plots)
    for i, plot in enumerate(plots):
        plt.subplot(row, cols, i + 1)
        plt.title(plot["title"])
        plt.imshow(plot["image"], cmap=plot.get("cmap", None))
        plt.axis("off")

    if score is not None:
        plt.text(
            0.5,
            1,
            f"Score: {score}",
            ha="center",
            va="top",
            transform=plt.gca().transAxes,
        )

    plt.show()


@run_app.command("adaptive_cfa_net", help="Run the Adaptive CFA Net method.")
def run_adaptive_cfa_net(
    image_path: Annotated[Path, typer.Argument(help=IMAGE_HELP)],
    output_folder: Annotated[
        Optional[Path],
        typer.Option(help=OUTPUT_FOLDER_HELP),
    ] = None,
    weights_path: Annotated[
        Optional[Path], typer.Option(help="Path to the weights.")
    ] = None,
    overlay: Annotated[bool, typer.Option(help=OVERLAY_HELP)] = False,
    show_plot: Annotated[bool, typer.Option(help=SHOW_PLOT_HELP)] = True,
    device: Annotated[str, typer.Option(help=DEVICE_HELP)] = "cpu",
):
    from photoholmes.methods.adaptive_cfa_net import (
        AdaptiveCFANet,
        adaptive_cfa_net_preprocessing,
    )

    image = read_image(image_path)
    model_input = adaptive_cfa_net_preprocessing(image=image.to(device))

    if weights_path is None:
        logger.info(
            "No weights provided, using default path `weights/adaptive_cfa_net/weights.pth`."  # noqa: E501
        )
        weights_path = Path("weights/adaptive_cfa_net/weights.pth")
        if not weights_path.exists():
            logger.error(
                "Weights not found. Please provide the correct path, or run "
                "`photoholmes download_weights adaptive_cfa_net` to download them."
            )
            return

    adaptive_cfa_net = AdaptiveCFANet("pretrained", weights=str(weights_path))
    adaptive_cfa_net.to_device(device)

    heatmap = adaptive_cfa_net.predict(**model_input)
    heatmap = heatmap.cpu()

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

        plt.imshow(heatmap.numpy())
        plt.savefig(output_folder / f"{image_path.stem}_adaptive_cfa_net_heatmap.png")
        logger.info(
            f"Heatmap saved to {output_folder / f'{image_path.stem}_adaptive_cfa_net_heatmap.png'}"  # noqa: E501
        )

    if show_plot:
        plots = [
            {"title": "Original Image", "image": image.permute(1, 2, 0).numpy()},
            {"title": "Heatmap", "image": heatmap.numpy(), "cmap": "hot"},
        ]
        if overlay:
            plots.append(
                {
                    "title": "Overlay",
                    "image": overlay_mask(
                        image.permute(1, 2, 0).numpy(), heatmap.numpy()
                    ),
                }
            )
        plot_results("Output of Adaptive CFA Net method", plots)


@run_app.command("catnet", help="Run the CatNet method.")
def run_catnet(
    image_path: Annotated[Path, typer.Argument(help=IMAGE_HELP)],
    output_folder: Annotated[
        Optional[Path],
        typer.Option(help=OUTPUT_FOLDER_HELP),
    ] = None,
    weights_path: Annotated[
        Optional[Path], typer.Option(help="Path to the weights.")
    ] = None,
    overlay: Annotated[bool, typer.Option(help=OVERLAY_HELP)] = False,
    show_plot: Annotated[bool, typer.Option(help=SHOW_PLOT_HELP)] = True,
    device: Annotated[str, typer.Option(help=DEVICE_HELP)] = "cpu",
):
    from photoholmes.methods.catnet import CatNet, catnet_preprocessing

    image = read_image(image_path)
    dct_coefficients, qtables = read_jpeg_data(str(image_path))

    model_input = catnet_preprocessing(
        image=image.to(device),
        dct_coefficients=dct_coefficients.to(device),
        qtables=qtables.to(device),
    )

    if weights_path is None:
        logger.info(
            "No weights provided, using default path `weights/catnet/weights.pth`."
        )
        weights_path = Path("weights/catnet/weights.pth")
        if not weights_path.exists():
            logger.error(
                "Weights not found. Please provide the correct path, or run "
                "`photoholmes download_weights catnet` to download them."
            )
            return

    catnet = CatNet("pretrained", weights=str(weights_path))
    catnet.to_device(device)

    heatmap, _ = catnet.predict(**model_input)
    heatmap = heatmap.cpu()

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

        plt.imshow(heatmap)
        plt.savefig(output_folder / f"{image_path.stem}_catnet_heatmap.png")
        logger.info(
            f"Heatmap saved to {output_folder / f'{image_path.stem}_catnet_heatmap.png'}"  # noqa: E501
        )

    if show_plot:
        plots = [
            {"title": "Original Image", "image": image.permute(1, 2, 0).numpy()},
            {"title": "Heatmap", "image": heatmap.numpy()},
        ]
        if overlay:
            plots.append(
                {
                    "title": "Overlay",
                    "image": overlay_mask(
                        image.permute(1, 2, 0).numpy(), heatmap.numpy()
                    ),
                }
            )
        plot_results("Output of CatNet method", plots)


@run_app.command("dq", help="Run the DQ method.")
def run_dq(
    image_path: Annotated[Path, typer.Argument(help=IMAGE_HELP)],
    output_folder: Annotated[
        Optional[Path], typer.Option(help=OUTPUT_FOLDER_HELP)
    ] = None,
    overlay: Annotated[bool, typer.Option(help=OVERLAY_HELP)] = False,
    show_plot: Annotated[bool, typer.Option(help=SHOW_PLOT_HELP)] = True,
):
    from photoholmes.methods.dq import DQ, dq_preprocessing

    image = read_image(image_path)
    dct_coefficients, _ = read_jpeg_data(str(image_path))
    model_input = dq_preprocessing(image=image, dct_coefficients=dct_coefficients)

    dq = DQ()

    heatmap = dq.predict(**model_input)

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

        plt.imshow(heatmap)
        plt.savefig(output_folder / f"{image_path.stem}_dq_heatmap.png")
        logger.info(
            f"Heatmap saved to {output_folder / f'{image_path.stem}_dq_heatmap.png'}"
        )

    if show_plot:
        plots = [
            {"title": "Original Image", "image": image.permute(1, 2, 0).numpy()},
            {"title": "Heatmap", "image": heatmap},
        ]
        if overlay:
            plots.append(
                {
                    "title": "Overlay heatmap",
                    "image": overlay_mask(image.permute(1, 2, 0).numpy(), heatmap),
                }
            )
        plot_results("Output of DQ method", plots)

    return


@run_app.command("exif_as_language", help="Run the Exif As Language method.")
def run_exif_as_language(
    image_path: Annotated[Path, typer.Argument(help=IMAGE_HELP)],
    output_folder: Annotated[
        Optional[Path],
        typer.Option(help=OUTPUT_FOLDER_HELP),
    ] = None,
    weights: Annotated[
        Optional[Path], typer.Option(help="Path to the weights.")
    ] = None,
    show_plot: Annotated[bool, typer.Option(help=SHOW_PLOT_HELP)] = True,
    overlay: Annotated[bool, typer.Option(help=OVERLAY_HELP)] = False,
    device: Annotated[str, typer.Option(help=DEVICE_HELP)] = "cpu",
):
    from photoholmes.methods.exif_as_language import (
        EXIFAsLanguage,
        exif_as_language_preprocessing,
    )

    image = read_image(image_path)
    model_input = exif_as_language_preprocessing(image=image.to(device))

    if weights is None:
        logger.info(
            "No weights provided, using default path `weights/exif_as_language/weights.pth`."  # noqa: E501
        )
        weights = Path("weights/exif_as_language/weights.pth")
        if not weights.exists():
            logger.error(
                "Weights not found. Please provide the correct path, or run "
                "`photoholmes download_weights exif_as_language` to download them."
            )
            return

    exif_as_language = EXIFAsLanguage(weights=str(weights))
    exif_as_language.to_device(device)

    heatmap, mask, score, *_ = exif_as_language.predict(**model_input)

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

        plt.imshow(mask)
        plt.savefig(output_folder / f"{image_path.stem}_exif_as_language_mask.png")
        logger.info(
            f"Mask saved to {output_folder / f'{image_path.stem}_exif_as_language_mask.png'}"  # noqa E501
        )

        plt.imshow(heatmap)
        plt.savefig(output_folder / f"{image_path.stem}_exif_as_language_heatmap.png")
        logger.info(
            f"Heatmap saved to {output_folder / f'{image_path.stem}_exif_as_language_heatmap.png'}"  # noqa E501
        )

        with open(
            output_folder / f"{image_path.stem}_exif_as_language_score.txt", "w"
        ) as f:
            f.write(f"score: {score}")
        logger.info(
            f"Score saved to {output_folder / f'{image_path.stem}_exif_as_language_score.txt'}"  # noqa E501
        )

    if show_plot:
        plots = [
            {"title": "Original Image", "image": image.permute(1, 2, 0).numpy()},
            {"title": "Mask", "image": mask, "cmap": "gray"},
            {"title": "Heatmap", "image": heatmap},
        ]
        if overlay:
            plots.insert(
                2,
                {
                    "title": "Overlay mask",
                    "image": overlay_mask(image.permute(1, 2, 0).numpy(), mask),
                },
            )
            plots.append(
                {
                    "title": "Overlay heatmap",
                    "image": overlay_mask(image.permute(1, 2, 0).numpy(), heatmap),
                }
            )
        plot_results("Output of Exif As Language method", plots)

    return


@run_app.command("focal", help="Run the Focal method.")
def run_focal(
    image_path: Annotated[Path, typer.Argument(help=IMAGE_HELP)],
    output_folder: Annotated[
        Optional[Path],
        typer.Option(help=OUTPUT_FOLDER_HELP),
    ] = None,
    vit_weights: Annotated[
        Optional[Path], typer.Option(help="Path to the ViT weights.")
    ] = None,
    hrnet_weights: Annotated[
        Optional[Path], typer.Option(help="Path to the HRNet weights.")
    ] = None,
    overlay: Annotated[bool, typer.Option(help=OVERLAY_HELP)] = False,
    show_plot: Annotated[bool, typer.Option(help=SHOW_PLOT_HELP)] = True,
    device: Annotated[str, typer.Option(help=DEVICE_HELP)] = "cpu",
):
    from photoholmes.methods.focal import Focal, focal_preprocessing

    image = read_image(image_path)
    model_input = focal_preprocessing(image=image.to(device))

    if vit_weights is None:
        logger.info(
            "No ViT weights provided, using default path `weights/focal/VIT_weights.pth`."  # noqa: E501
        )
        vit_weights = Path("weights/focal/VIT_weights.pth")
        if not vit_weights.exists():
            logger.error(
                "ViT weights not found. Please provide the correct path, or run "
                "`photoholmes download_weights focal` to download them."
            )
            return
    if hrnet_weights is None:
        logger.info(
            "No HRNet weights provided, using default path `weights/focal/HRNET_weights.pth`."  # noqa: E501
        )
        hrnet_weights = Path("weights/focal/HRNET_weights.pth")
        if not hrnet_weights.exists():
            logger.error(
                "HRNet weights not found. Please provide the correct path, or run "
                "`photoholmes download_weights focal` to download them."
            )
            return

    focal = Focal(weights={"ViT": str(vit_weights), "HRNet": str(hrnet_weights)})
    focal.to_device(device)

    mask = focal.predict(**model_input).cpu()

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

        plt.savefig(output_folder / f"{image_path.stem}_focal_mask.png")
        logger.info(
            f"Mask saved to {output_folder / f'{image_path.stem}_focal_mask.png'}"
        )

    if show_plot:
        plots = [
            {"title": "Original Image", "image": image.permute(1, 2, 0).numpy()},
            {"title": "Mask", "image": mask.numpy()},
        ]
        if overlay:
            plots.append(
                {
                    "title": "Overlay",
                    "image": overlay_mask(image.permute(1, 2, 0).numpy(), mask.numpy()),
                }
            )
        plot_results(
            "Output of Focal method",
            plots,
        )

    return


@run_app.command("noisesniffer", help="Run the Noisesniffer method.")
def run_noisesniffer(
    image_path: Annotated[Path, typer.Argument(help=IMAGE_HELP)],
    output_folder: Annotated[
        Optional[Path], typer.Option(help=OUTPUT_FOLDER_HELP)
    ] = None,
    overlay: Annotated[bool, typer.Option(help=OVERLAY_HELP)] = False,
    show_plot: Annotated[bool, typer.Option(help=SHOW_PLOT_HELP)] = True,
):
    from photoholmes.methods.noisesniffer import (
        Noisesniffer,
        noisesniffer_preprocessing,
    )

    image = read_image(image_path)
    model_input = noisesniffer_preprocessing(image=image)

    noisesniffer = Noisesniffer()

    mask, _ = noisesniffer.predict(**model_input)

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

        plt.imshow(mask)
        plt.savefig(output_folder / f"{image_path.stem}_noisesniffer_mask.png")
        logger.info(
            f"Mask saved to {output_folder / f'{image_path.stem}_noisesniffer_mask.png'}"  # noqa: E501
        )

    if show_plot:
        plots = [
            {"title": "Original Image", "image": image.permute(1, 2, 0).numpy()},
            {"title": "Mask", "image": mask},
        ]
        if overlay:
            plots.append(
                {
                    "title": "Overlay",
                    "image": overlay_mask(image.permute(1, 2, 0).numpy(), mask),
                }
            )
        plot_results("Output of Noisesniffer method", plots)


@run_app.command("psccnet", help="Run the PSCCNet method.")
def run_psccnet(
    image_path: Annotated[Path, typer.Argument(help=IMAGE_HELP)],
    output_folder: Annotated[
        Optional[Path], typer.Option(help=OUTPUT_FOLDER_HELP)
    ] = None,
    weights_folder: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to the weights folder. Inside the folder, the weights should be named `ClsNet.pth`, `FENet.pth` and `SegNet.pth`."  # noqa: E501"
        ),
    ] = None,
    overlay: Annotated[bool, typer.Option(help=OVERLAY_HELP)] = False,
    show_plot: Annotated[bool, typer.Option(help=SHOW_PLOT_HELP)] = True,
    device: Annotated[str, typer.Option(help=DEVICE_HELP)] = "cpu",
):
    from photoholmes.methods.psccnet import PSCCNet, psccnet_preprocessing

    image = read_image(image_path)
    model_input = psccnet_preprocessing(image=image.to(device))

    if weights_folder is None:
        logger.info("No weights provided, using default path `weights/psccnet/`.")
        weights_folder = Path("weights/psccnet/")
        if not weights_folder.exists():
            logger.error(
                "Weights not found. Please provide the correct path, or run "
                "`photoholmes download_weights psccnet` to download them."
            )
            return

    cls_net_path = weights_folder / "ClsNet.pth"
    if not cls_net_path.exists():
        logger.error(
            "FENet.pth not found. Make the sure ClsNet.pth is in the folder, or run "
            "`photoholmes download_weights psccnet` to download them."
        )
        return
    fe_net_path = weights_folder / "FENet.pth"
    if not fe_net_path.exists():
        logger.error(
            "FENet.pth not found. Make the sure FENet.pth is in the folder, or run "
            "`photoholmes download_weights psccnet` to download them."
        )
        return
    seg_net_path = weights_folder / "SegNet.pth"
    if not seg_net_path.exists():
        logger.error(
            "SegNet.pth not found. Make the sure SegNet.pth is in the folder, or run "
            "`photoholmes download_weights psccnet` to download them."
        )
        return

    psccnet = PSCCNet(
        {
            "FENet": str(fe_net_path),
            "SegNet": str(seg_net_path),
            "ClsNet": str(cls_net_path),
        },
        device=device,
    )

    heatmap, detection = psccnet.predict(**model_input)
    heatmap = heatmap.cpu()
    detection = detection.cpu()

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

        plt.imshow(heatmap)
        plt.savefig(output_folder / f"{image_path.stem}_psccnet_heatmap.png")
        logger.info(
            f"Heatmap saved to {output_folder / f'{image_path.stem}_psccnet_heatmap.png'}"  # noqa: E501
        )
        with open(output_folder / f"{image_path.stem}_psccnet_detection.txt", "w") as f:
            f.write(f"score: {detection.item()}")
        logger.info(
            f"Detection score saved to {output_folder / f'{image_path.stem}_psccnet_detection.txt'}"  # noqa: E501
        )

    if show_plot:
        plots = [
            {"title": "Original Image", "image": image.permute(1, 2, 0).numpy()},
            {"title": "Heatmap", "image": heatmap.numpy()},
        ]
        if overlay:
            plots.append(
                {
                    "title": "Overlay heatmap",
                    "image": overlay_mask(
                        image.permute(1, 2, 0).numpy(), heatmap.numpy()
                    ),
                }
            )
        plot_results("Output of PSCCNet method", plots)


@run_app.command("splicebuster", help="Run the Splicebuster method.")
def run_splicebuster(
    image_path: Annotated[Path, typer.Argument(help=IMAGE_HELP)],
    output_folder: Annotated[
        Optional[Path], typer.Option(help=OUTPUT_FOLDER_HELP)
    ] = None,
    overlay: Annotated[bool, typer.Option(help=OVERLAY_HELP)] = False,
    show_plot: Annotated[bool, typer.Option(help=SHOW_PLOT_HELP)] = True,
):
    from photoholmes.methods.splicebuster import (
        Splicebuster,
        splicebuster_preprocessing,
    )

    image = read_image(str(image_path))
    model_input = splicebuster_preprocessing(image=image)

    splicebuster = Splicebuster()

    heatmap = splicebuster.predict(**model_input)

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

        plt.imshow(heatmap)
        plt.savefig(output_folder / f"{image_path.stem}_splicebuster_heatmap.png")
        logger.info(
            f"Heatmap saved to {output_folder / f'{image_path.stem}_splicebuster_heatmap.png'}"  # noqa: E501
        )

    if show_plot:
        plots = [
            {"title": "Original Image", "image": image.permute(1, 2, 0).numpy()},
            {"title": "Heatmap", "image": heatmap},
        ]
        if overlay:
            plots.append(
                {
                    "title": "Overlay heatmap",
                    "image": overlay_mask(image.permute(1, 2, 0).numpy(), heatmap),
                }
            )
        plot_results("Output of Splicebuster method", plots)


@run_app.command("trufor", help="Run the TruFor method.")
def run_trufor(
    image_path: Annotated[Path, typer.Argument(help=IMAGE_HELP)],
    output_folder: Annotated[
        Optional[Path], typer.Option(help=OUTPUT_FOLDER_HELP)
    ] = None,
    weights_path: Annotated[
        Optional[Path], typer.Option(help="Path to the weights.")
    ] = None,
    overlay: Annotated[bool, typer.Option(help=OVERLAY_HELP)] = False,
    show_plot: Annotated[bool, typer.Option(help=SHOW_PLOT_HELP)] = True,
    device: Annotated[str, typer.Option(help=DEVICE_HELP)] = "cpu",
):
    from photoholmes.methods.trufor import TruFor, trufor_preprocessing

    image = read_image(image_path)
    model_input = trufor_preprocessing(image=image.to(device))

    if weights_path is None:
        logger.info(
            "No weights provided, using default path `weights/trufor/trufor.pth.tar`."
        )
        weights_path = Path("weights/trufor/trufor.pth.tar")
        if not weights_path.exists():
            logger.error(
                "Weights not found. Please provide the correct path, or run "
                "`photoholmes download_weights trufor` to download them."
            )
            return

    trufor = TruFor(weights=str(weights_path))
    trufor.to_device(device)

    heatmap, confidence, detection, _ = trufor.predict(**model_input)
    heatmap = heatmap.cpu()
    if confidence is not None:
        confidence = confidence.cpu()
    if detection is not None:
        detection = detection.cpu()

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

        plt.imshow(heatmap)
        plt.savefig(output_folder / f"{image_path.stem}_trufor_heatmap.png")
        logger.info(
            f"Heatmap saved to {output_folder / f'{image_path.stem}_trufor_heatmap.png'}"  # noqa: E501
        )
        if confidence is not None:
            plt.imshow(confidence.numpy())
            plt.savefig(output_folder / f"{image_path.stem}_trufor_confidence.png")
            logger.info(
                f"Confidence saved to {output_folder / f'{image_path.stem}_trufor_confidence.png'}"  # noqa: E501
            )
        if detection is not None:
            with open(
                output_folder / f"{image_path.stem}_trufor_detection.txt", "w"
            ) as f:
                f.write(f"score: {detection.item()}")

    if show_plot:
        plots = [
            {"title": "Original Image", "image": image.permute(1, 2, 0).numpy()},
            {"title": "Heatmap", "image": heatmap.numpy()},
        ]

        if confidence is not None:
            cool_heatmap = (heatmap * confidence).numpy()
            plots.extend(
                [
                    {"title": "Confidence", "image": confidence.numpy()},
                    {"title": "Heatmap w/confidence", "image": cool_heatmap},
                ]
            )
        if overlay:
            plots.extend(
                [
                    {
                        "title": "Overlay heatmap",
                        "image": overlay_mask(
                            image.permute(1, 2, 0).numpy(), heatmap.numpy()
                        ),
                    },
                    {
                        "title": "Overlay heatmap w/confidence",
                        "image": overlay_mask(
                            image.permute(1, 2, 0).numpy(), cool_heatmap
                        ),
                    },
                ]
            )

        plot_results("Output of TruFor method", plots)
    return


@run_app.command("zero", help="Run the Zero method.")
def run_zero(
    image_path: Annotated[Path, typer.Argument(help=IMAGE_HELP)],
    output_folder: Annotated[
        Optional[Path], typer.Option(help=OUTPUT_FOLDER_HELP)
    ] = None,
    missing_grids: Annotated[
        bool, typer.Option(help="Include missing grid mask in plot.")
    ] = True,
    overlay: Annotated[bool, typer.Option(help=OVERLAY_HELP)] = False,
    show_plot: Annotated[bool, typer.Option(help=SHOW_PLOT_HELP)] = True,
):
    import numpy as np

    from photoholmes.methods.zero import Zero, zero_preprocessing

    image = read_image(image_path)
    model_input = zero_preprocessing(image=image)

    zero = Zero(missing_grids=missing_grids)

    logger.info("Running Zero method")
    _, mask, votes, missing_grids_mask = zero.predict(**model_input)

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

        plt.imshow(mask)
        plt.savefig(output_folder / f"{image_path.stem}_zero_forgery_mask.png")
        logger.info(
            f"Mask saved to {output_folder / f'{image_path.stem}_zero_mask.png'}"
        )
        np.save(output_folder / f"{image_path.stem}_zero_votes.npy", votes)
        logger.info(
            f"Votes saved to {output_folder / f'{image_path.stem}_zero_votes.npy'}"
        )
        if missing_grids_mask:
            plt.imshow(missing_grids_mask)
            plt.savefig(output_folder / f"{image_path.stem}_zero_missing_grids.png")
            logger.info(
                f"Votes saved to {output_folder / f'{image_path.stem}_zero_missing_grids.png'}"  # noqa: E501
            )

    if show_plot:
        plots = [
            {"title": "Original Image", "image": image.permute(1, 2, 0).numpy()},
            {"title": "Mask", "image": mask},
            {"title": "Votes", "image": votes},
        ]
        if missing_grids_mask is not None:
            plots.extend(
                [
                    {"title": "Missing grids", "image": missing_grids_mask},
                    {
                        "title": "Mask + Missing grids",
                        "image": np.logical_or(mask, missing_grids_mask),
                    },
                ]
            )
        if overlay:
            plots.append(
                {
                    "title": "Mask Overlay",
                    "image": overlay_mask(image.permute(1, 2, 0).numpy(), mask),
                }
            )
            if missing_grids_mask is not None:
                plots.append(
                    {
                        "title": "Missing grids Overlay",
                        "image": overlay_mask(
                            image.permute(1, 2, 0).numpy(), missing_grids_mask
                        ),
                    }
                )
                plots.append(
                    {
                        "title": "Mask + Missing grids Overlay",
                        "image": overlay_mask(
                            image.permute(1, 2, 0).numpy(),
                            np.logical_or(missing_grids_mask, mask).astype(int),
                        ),
                    }
                )
        plot_results("Output of Zero method", plots)
