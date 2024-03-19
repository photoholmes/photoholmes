from pathlib import Path
from typing import Annotated, Optional

import typer
from typer import Typer

app = Typer(name="adapt_weights", help="Adapt weights for a photoholmes method")


@app.command("exif_as_language", help="adapt the exif as language clip's weights")
def adapt_exif_weights(
    weights_path: Annotated[
        Path,
        typer.Argument(help="path to the original weights", case_sensitive=False),
    ],
    out_path: Annotated[
        Optional[Path], typer.Argument(help="path to save the weights to")
    ] = None,
):
    from photoholmes.methods.exif_as_language.prune_original_weights import (
        prune_original_weights,
    )

    if out_path is None:
        out_path = Path("/".join(weights_path.parts[:-1]))
        out_path = out_path / f"{weights_path.stem}_adapted{weights_path.suffix}"

    prune_original_weights(weights_path, str(out_path))
