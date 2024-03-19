import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from torchmetrics import Metric, MetricCollection
from tqdm import tqdm

from photoholmes.datasets.base import BaseDataset
from photoholmes.methods.base import BaseMethod, BenchmarkOutput

logging.basicConfig(format="%(levelname)s - %(message)s")
IO_MESSAGE = 11
logging.addLevelName(IO_MESSAGE, "IO_MESSAGE")


def io_message(self, message, *args, **kws):
    if self.isEnabledFor(IO_MESSAGE):
        self._log(IO_MESSAGE, message, args, **kws)


logging.Logger.io_message = io_message
log = logging.getLogger(__name__)

verbose_dict = {
    0: logging.WARNING,
    1: logging.INFO,
    2: IO_MESSAGE,
}


class Benchmark:
    """
    Benchmark class for evaluating the performance of image processing methods.

    Attributes:
        save_method_outputs (bool): Whether to save the method outputs.
        save_extra_outputs (bool): Whether to save extra outputs.
        save_metrics_flag (bool): Whether to save metrics.
        output_path (Path): Path to the output folder.
        device (torch.device): Device for computation.
        use_existing_output (bool): Whether to use existing saved outputs.
        verbose (int): Verbosity level.

    Methods:
        run(method, dataset, metrics):
            Run the benchmark using the specified method, dataset, and metrics.
    """

    def __init__(
        self,
        save_method_outputs: bool = True,
        save_extra_outputs: bool = False,
        save_metrics: bool = True,
        output_folder: str = "output/",
        device: str = "cpu",
        use_existing_output: bool = True,
        verbose: Literal[0, 1, 2] = 1,
    ):
        """
        Args:
            save_method_outputs (bool): Whether to save the method outputs.
                Default is True.
            save_extra_outputs (bool): Whether to save extra outputs.
                Default is False.
            save_metrics (bool): Whether to save metrics. Default is True.
            output_folder (str): Folder to save outputs. Default is "output/".
            device (str): Device for computation (e.g., "cpu" or "cuda").
                Default is "cpu".
            use_existing_output (bool): Whether to use existing saved outputs.
                Default is True.
            verbose (Literal[0, 1, 2]): Verbosity level (0, 1, or 2). Default is 1.
        """
        self.save_method_outputs = save_method_outputs
        self.save_extra_outputs = save_extra_outputs
        self.save_metrics_flag = save_metrics
        self.output_path = Path(output_folder)
        self.use_existing_output = use_existing_output
        self.verbose = verbose

        if self.verbose not in verbose_dict:
            log.warning(
                f"Invalid verbose level '{self.verbose}'. "
                f"Using default verbose level '1'."
            )
            self.verbose = 1

        log.setLevel(verbose_dict[self.verbose])

        if device.startswith("cuda") and not torch.cuda.is_available():
            log.warning(
                f"Requested device '{device}' is not available. Falling back to 'cpu'."
            )
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self._mask = False
        self._heatmap = False
        self._detection = False

    def run(
        self,
        method: BaseMethod,
        dataset: BaseDataset,
        metrics: Union[MetricCollection, List[Metric]],
    ):
        """
        Run the benchmark using the specified method, dataset, and metrics.

        Args:
            method (BaseMethod): The method to evaluate.
            dataset (BaseDataset): Dataset to run the evaluation on.
            metrics (MetricCollection): Collection of metrics to compute.

        Returns:
            dict: Computed metrics for the benchmark.
        """
        log.info(f"Using device: {self.device}")
        if method.device != self.device:
            method.to_device(self.device)

        if isinstance(metrics, list):
            metrics = MetricCollection(metrics)

        output_path = (
            self.output_path
            / method.__class__.__name__.lower()
            / dataset.__class__.__name__.lower()
        )
        self._print_setup_message(method, dataset, metrics, output_path)

        heatmap_metrics = metrics.clone(prefix="heatmap").to(
            self.device, dtype=torch.float32
        )
        mask_metrics = metrics.clone(prefix="mask").to(self.device, dtype=torch.float32)
        detection_metrics = metrics.clone(prefix="detection").to(
            self.device, dtype=torch.float32
        )

        image_count = 0
        for data, mask, image_name in tqdm(dataset, desc="Processing Images"):  # type: ignore
            output = None
            if self.use_existing_output:
                output = self._load_existing_output(output_path, image_name)

            if output is None:
                data_on_device = self._dict_to_device(data)
                output = method.benchmark(**data_on_device)

                if self.save_method_outputs:
                    self._save_predicted_output(output_path, image_name, output)

            mask = mask.to(self.device)
            if output["detection"] is not None:
                if output["detection"].ndim == 2:
                    output["detection"] = output["detection"].squeeze(0)

                detection_gt = (
                    torch.tensor(int(torch.any(mask)))
                    .unsqueeze(0)
                    .to(self.device, dtype=torch.int32)
                )

                detection_metrics.update(output["detection"], detection_gt)
                self._detection = True

            if output["mask"] is not None:
                if output["mask"].ndim == 3:
                    output["mask"] = output["mask"].squeeze(0)

                mask_metrics.update(output["mask"], mask)
                self._mask = True

            if output["heatmap"] is not None:
                if output["heatmap"].ndim == 3:
                    output["heatmap"] = output["heatmap"].squeeze(0)

                heatmap_metrics.update(output["heatmap"], mask)
                self._heatmap = True

            image_count += 1

        log.info("-" * 80)
        log.info("-" * 80)

        if self.save_metrics_flag:
            tampered = (
                "tampered_only" if dataset.tampered_only else "tampered_and_pristine"
            )
            timestamp = time.strftime("%Y%m%d_%H:%M")

            report_id = f"{timestamp}_{tampered}"
            if self._heatmap:
                log.info("     - Saving heatmap metrics")
                self._save_metrics(
                    output_path=output_path,
                    metrics=heatmap_metrics,
                    report_id=report_id,
                    total_images=image_count,
                )
            else:
                log.info("     - No heatmap metrics to save")

            if self._mask:
                log.info("     - Saving mask metrics")
                self._save_metrics(
                    output_path=output_path,
                    metrics=mask_metrics,
                    report_id=report_id,
                    total_images=image_count,
                )
            else:
                log.info("     - No mask metrics to save")

            if self._detection:
                log.info("     - Saving detection metrics")
                self._save_metrics(
                    output_path=output_path,
                    metrics=detection_metrics,
                    report_id=report_id,
                    total_images=image_count,
                )
            else:
                log.info("     - No detection metrics to save")
        else:
            log.info("     - Not saving metrics")

        log.info("-" * 80)
        log.info("-" * 80)
        log.info("Benchmark finished")
        log.info("-" * 80)
        log.info("-" * 80)

        metrics_return = {}
        if self._heatmap:
            metrics_return["heatmap"] = heatmap_metrics.compute()
        if self._mask:
            metrics_return["mask"] = mask_metrics.compute()
        if self._detection:
            metrics_return["detection"] = detection_metrics.compute()

        return metrics_return

    def _print_setup_message(
        self,
        method: BaseMethod,
        dataset: BaseDataset,
        metrics: MetricCollection,
        output_path: Path,
    ):
        """
        Print the benchmark setup message.

        Args:
            method (BaseMethod): The method to evaluate.
            dataset (BaseDataset): Dataset to run the evaluation on.
            metrics (MetricCollection): Collection of metrics to compute.
            output_path (Path): Path to the output folder.
        """
        log.info("-" * 80)
        log.info("-" * 80)
        log.info("Running the benchmark")
        log.info("-" * 80)
        log.info("-" * 80)
        log.info("Benchmark configuration:")
        log.info(f"    Method: {method.__class__.__name__}")
        log.info(f"    Dataset: {dataset.__class__.__name__}")
        log.info("    Metrics:")
        for metric in metrics:
            log.info(f"       - {metric}")
        log.info(f"    Output path: {output_path}")
        log.info(f"    Save method outputs: {self.save_method_outputs}")
        log.info(f"    Save metrics: {self.save_metrics_flag}")
        log.info(f"    Device: {self.device}")
        log.info(f"    Load existing outputs: {self.use_existing_output}")
        log.info(f"    Verbosity: {logging._levelToName[verbose_dict[self.verbose]]}")
        log.info("-" * 80)
        log.info("-" * 80)

    def _dict_to_device(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move dict items to the benchmark's device.

        Args:
            data (Dict[str, Any]): Data to move to the benchmark's device.

        Returns:
            Dict[str, Any]: Data moved to the benchmark's device.
        """
        return {
            key: (
                value.to(self.device, dtype=torch.float32)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in data.items()
        }

    def _save_metrics(
        self,
        output_path: Path,
        metrics: MetricCollection,
        report_id: str,
        total_images: int,
    ):
        """
        Save predicted outputs for an image.

        Args:
            output_path (Path): Path to the output folder.
            metrics (MetricCollection): Collection of metrics to compute.
            report_id (str): ID for the report.
            total_images (int): Total number of images processed.

        """
        metrics_path = output_path / "metrics" / report_id
        os.makedirs(metrics_path, exist_ok=True)

        metric_compute = metrics.compute()
        torch.save(metrics.state_dict(), metrics_path / f"{metrics.prefix}_state.pt")

        metric_report: Dict[str, Any] = {}
        for key, value in metric_compute.items():
            if isinstance(value, torch.Tensor) and value.dim() == 0:
                metric_report[key] = float(value)
            elif isinstance(value, tuple) and all(
                isinstance(v, torch.Tensor) for v in value
            ):
                metric_report[key] = [v.tolist() for v in value]
            elif (
                isinstance(value, int)
                or isinstance(value, float)
                or isinstance(value, str)
            ):
                metric_report[key] = value
            else:
                log.warning(f"Skipping metric '{key}' of type '{type(value)}'")

        report = {
            "metrics": metric_report,
            "total_images": total_images,
            "type": metrics.prefix,
        }

        with open(metrics_path / f"{metrics.prefix}_report.json", "w") as f:
            json.dump(report, f)

    def _save_predicted_output(
        self, output_path: Path, image_name: str, output: BenchmarkOutput
    ):
        """
        Save predicted outputs for an image.

        Args:
            output_path (Path): Path to the output folder.
            image_name (str): Name of the processed image.
            output (BenchmarkOutput): Output to save.
        """
        image_save_path = output_path / "outputs" / image_name
        os.makedirs(image_save_path, exist_ok=True)

        output_dict = {}
        if output["heatmap"] is not None:
            output_dict["heatmap"] = output["heatmap"].cpu().numpy()
        if output["mask"] is not None:
            output_dict["mask"] = output["mask"].cpu().numpy()
        if output["detection"] is not None:
            output_dict["detection"] = output["detection"].cpu().numpy()
        np.savez_compressed(image_save_path / "output", **output_dict)

        if "extra_outputs" in output:
            extra_outputs_arrays = {}
            extra_outputs_other = {}

            for key, value in output["extra_outputs"].items():
                if isinstance(value, (torch.Tensor)):
                    extra_outputs_arrays[key] = value.cpu()
                elif isinstance(value, (list, np.ndarray)):
                    extra_outputs_arrays[key] = value
                else:
                    extra_outputs_other[key] = value

            if self.save_extra_outputs:
                np.savez_compressed(
                    image_save_path / "extra_outputs_arrays", **extra_outputs_arrays
                )
                with open(image_save_path / "extra_outputs_other.json", "w") as f:
                    json.dump(extra_outputs_other, f)

        log.io_message(f"Output for image '{image_name}' saved.")

    def _load_existing_output(
        self, output_path: Path, image_name: str
    ) -> Optional[BenchmarkOutput]:
        """
        Load existing output for a given image.

        Args:
            output_path (Path): Path to the output folder.
            image_name (str): Name of the processed image.

        Returns:
            Optional[BenchmarkOutput]: Loaded output if available, else None.
        """
        output_path = output_path / "outputs"
        if not os.path.exists(output_path):
            return None

        if os.path.exists(output_path / image_name / "output.npz"):
            log.io_message(f"Loading existing output for image '{image_name}'")

            prior_output = np.load(
                output_path / image_name / "output.npz",
                allow_pickle=True,
            )
            output: BenchmarkOutput = {
                "heatmap": (
                    torch.tensor(prior_output["heatmap"], device=self.device)
                    if "heatmap" in prior_output
                    else None
                ),
                "mask": (
                    torch.tensor(prior_output["mask"], device=self.device)
                    if "mask" in prior_output
                    else None
                ),
                "detection": (
                    torch.tensor(prior_output["detection"], device=self.device)
                    if "detection" in prior_output
                    else None
                ),
            }
            return output

        log.io_message(f"No existing output found for image '{image_name}'.")
        return None
