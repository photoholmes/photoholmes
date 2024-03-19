import logging
from pathlib import Path

import typer
from typing_extensions import Annotated

from photoholmes.methods.registry import MethodRegistry

from .adapt_weights import app as adapt_weights_app
from .run import run_app

logging.basicConfig()
logger = logging.getLogger(__name__)

app = typer.Typer()
app.add_typer(run_app, name="run", help="Run a method on an image.")


@app.command(name="download_weights", help="Automatic weight download for a method")
def run_download_weights(
    method: Annotated[
        MethodRegistry,
        typer.Argument(help="method", case_sensitive=False),
    ],
    weight_folder: Annotated[Path, typer.Option(help="Path to weight folder.")] = Path(
        "weights"
    ),
):
    if method == MethodRegistry.PSCCNET:
        from .download_weights import download_psccnet_weights

        download_psccnet_weights(weight_folder / "psccnet")
    elif method == MethodRegistry.EXIF_AS_LANGUAGE:
        from .download_weights import download_exif_weights

        download_exif_weights(weight_folder / "exif_as_language")
    elif method == MethodRegistry.ADAPTIVE_CFA_NET:
        from .download_weights import download_adaptive_cfa_net_weights

        download_adaptive_cfa_net_weights(weight_folder / "adaptive_cfa_net")
    elif method == MethodRegistry.CATNET:
        from .download_weights import download_catnet_weights

        logger.warning(
            "CatNet weights are under a non-commercial license. "
            "See https://github.com/mjkwon2021/CAT-Net/tree/main?tab=readme-ov-file#licence for more information."  # noqa: E501
        )
        r = input("Press yes if you agree to the license [yes/no]: ")
        while r.lower() not in ["no", "n", "yes", "y"]:
            r = input("Press yes if you agree to the license [yes/no]: ")
        if r.lower() in ["no", "n"]:
            logger.warning("You must agree to the license to download the weights.")
            return
        download_catnet_weights(weight_folder / "catnet")
    elif method == MethodRegistry.TRUFOR:
        from .download_weights import download_trufor_weights

        logger.warning(
            "TruFor weights are under a non-commercial license. See https://github.com/grip-unina/TruFor/blob/main/test_docker/LICENSE.txt for more information."  # noqa: E501
        )
        r = input("Press yes if you agree to the license [yes/no]: ")
        while r.lower() not in ["no", "n", "yes", "y"]:
            r = input("Press yes if you agree to the license [yes/no]: ")
        if r.lower() in ["no", "n"]:
            logger.warning("You must agree to the license to download the weights.")
            return

        download_trufor_weights(weight_folder / "trufor")
    elif method == MethodRegistry.FOCAL:
        from .download_weights import download_focal_weights

        download_focal_weights(weight_folder / "focal")
    else:
        logging.info(
            "No weights available for this method. Check the method README "
            "for more information."
        )


app.add_typer(run_app)
app.add_typer(adapt_weights_app)


def cli():
    app()
