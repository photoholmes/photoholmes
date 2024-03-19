import logging
from pathlib import Path
from typing import Any, Dict, List, Literal

import typer
from pydantic import BaseModel
from tqdm import tqdm

from photoholmes.datasets.registry import DatasetRegistry
from photoholmes.methods.registry import MethodRegistry
from photoholmes.metrics.registry import MetricRegistry
from photoholmes.utils.generic import load_yaml

app = typer.Typer()
logger = logging.getLogger(__name__)


def run_benchmark(
    method_name: MethodRegistry,
    method_config: str | dict,
    dataset_name: DatasetRegistry,
    dataset_path: str,
    metrics: List[str],
    tampered_only: bool = False,
    save_method_outputs: bool = False,
    output_folder: str = "output/",
    device: str = "cpu",
):
    from photoholmes.benchmark import Benchmark
    from photoholmes.datasets.factory import DatasetFactory
    from photoholmes.methods.factory import MethodFactory
    from photoholmes.metrics.factory import MetricFactory

    # Load method and preprocessing
    # You can add a custom method in this cli by doing something like:
    # if method_name == "my_custom_method":
    #     from my_custom_module import MyCustomMethod, CustomPreprocessing
    #     method = MyCustomMethod()
    #     method.to_device(device)
    #     preprocessing = CustomPreprocessing()
    # else:
    method, preprocessing = MethodFactory.load(
        method_name=method_name, config=method_config
    )
    method.to_device(device)

    # Load dataset
    dataset = DatasetFactory.load(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tampered_only=tampered_only,
        preprocessing_pipeline=preprocessing,
    )

    metrics_objects = MetricFactory.load(metrics)

    # Create Benchmark
    benchmark = Benchmark(
        save_method_outputs=save_method_outputs,
        output_folder=output_folder,
        device=device,
    )

    # Run Benchmark
    benchmark.run(
        method=method,
        dataset=dataset,
        metrics=metrics_objects,
    )


@app.command()
def main(
    method_name: MethodRegistry = typer.Option(..., help="Name of the method to use."),
    method_config: str = typer.Option(
        None, help="Path to the configuration file for the method."
    ),
    dataset_name: DatasetRegistry = typer.Option(..., help="Name of the dataset."),
    dataset_path: str = typer.Option(..., help="Path to the dataset."),
    metrics: str = typer.Option(
        ..., "--metrics", help="Space-separated list of metrics to use."
    ),
    tampered_only: bool = typer.Option(False, help="Process tampered images only."),
    save_method_outputs: bool = typer.Option(False, help="Save the output."),
    output_folder: str = typer.Option("output/", help="Path to save the outputs."),
    device: str = typer.Option("cpu", help="Device to use."),
):
    """
    Run the Benchmark for image tampering detection.
    """

    # Load metrics
    metrics_list = metrics.split()
    run_benchmark(
        method_name=method_name,
        method_config=method_config,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        metrics=metrics_list,
        tampered_only=tampered_only,
        save_method_outputs=save_method_outputs,
        output_folder=output_folder,
        device=device,
    )


class DatasetSpec(BaseModel):
    name: DatasetRegistry
    path: str
    tampered_only: bool


class BenchmarkConfig(BaseModel):
    method_name: MethodRegistry
    method_config: Dict[str, Any]
    datasets: List[DatasetSpec]
    metrics: List[MetricRegistry]
    save_method_outputs: bool = True
    save_metrics: bool = True
    output_folder: str = "output/"
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    use_existing_output: bool = True
    verbose: Literal[0, 1, 2] = 1


@app.command("from_config")
def run_from_config(
    config_path: str = typer.Argument(..., help="Path to the configuration file.")
):
    bench_config = BenchmarkConfig(**load_yaml(config_path))

    from photoholmes.benchmark import Benchmark
    from photoholmes.datasets.factory import DatasetFactory
    from photoholmes.methods.factory import MethodFactory
    from photoholmes.metrics.factory import MetricFactory

    # Load method and preprocessing
    method, preprocessing = MethodFactory.load(
        method_name=bench_config.method_name,
        config=bench_config.method_config,
    )
    method.to_device(bench_config.device)

    # Load datasets
    datasets = []
    for d in tqdm(bench_config.datasets, desc="Setting up datasets"):
        dataset = DatasetFactory.load(
            dataset_name=d.name,
            dataset_path=d.path,
            tampered_only=d.tampered_only,
            preprocessing_pipeline=preprocessing,
        )
        if len(dataset) == 0:
            logger.warning(f"Dataset {d.name} is empty.")
            continue
        datasets.append(dataset)

    metrics_objects = MetricFactory.load(bench_config.metrics)

    # Create Benchmark
    benchmark = Benchmark(
        save_method_outputs=bench_config.save_method_outputs,
        save_metrics=bench_config.save_metrics,
        output_folder=bench_config.output_folder,
        device=bench_config.device,
        use_existing_output=bench_config.use_existing_output,
        verbose=bench_config.verbose,
    )

    # Run Benchmark
    for dataset in datasets:
        benchmark.run(
            method=method,
            dataset=dataset,
            metrics=metrics_objects,
        )


@app.command("process_outputs")
def process_output(
    method: MethodRegistry = typer.Argument(),
    outputs_dir: Path = typer.Argument(),
    upload_dir: Path = typer.Argument(),
):
    from prepare_outputs import process_output_folder

    process_output_folder(method.value, outputs_dir, upload_dir)


if __name__ == "__main__":
    app()
