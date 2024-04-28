from typing import List, Literal, Optional, Union

from photoholmes.datasets.base import BaseDataset
from photoholmes.datasets.registry import DatasetRegistry
from photoholmes.preprocessing.pipeline import PreProcessingPipeline


class DatasetFactory:
    @staticmethod
    def load(
        dataset_name: Union[str, DatasetRegistry],
        dataset_path: str,
        preprocessing_pipeline: Optional[PreProcessingPipeline] = None,
        load: List[
            Literal[
                "image",
                "dct_coefficients",
                "qtables",
            ]
        ] = [
            "image",
            "dct_coefficients",
            "qtables",
        ],
        tampered_only: bool = False,
    ) -> BaseDataset:
        """
        Instantiates and returns a dataset object corresponding to the specified
        dataset name.

        """
        if isinstance(dataset_name, str):
            dataset_name = DatasetRegistry(dataset_name.lower())

        match dataset_name:
            case DatasetRegistry.COLUMBIA:
                from photoholmes.datasets.columbia import ColumbiaDataset

                return ColumbiaDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )
            case DatasetRegistry.COLUMBIA_OSN:
                from photoholmes.datasets.osn import ColumbiaOSNDataset

                return ColumbiaOSNDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=True,
                )
            case DatasetRegistry.COLUMBIA_WEBP:
                from photoholmes.datasets.columbia_webp import ColumbiaWebPDataset

                return ColumbiaWebPDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )
            case DatasetRegistry.COVERAGE:
                from photoholmes.datasets.coverage import CoverageDataset

                return CoverageDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )
            case DatasetRegistry.REALISTIC_TAMPERING:
                from photoholmes.datasets.realistic_tampering import (
                    RealisticTamperingDataset,
                )

                return RealisticTamperingDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )
            case DatasetRegistry.REALISTIC_TAMPERING_WEBP:
                from photoholmes.datasets.realistic_tampering_webp import (
                    RealisticTamperingWebPDataset,
                )

                return RealisticTamperingWebPDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )

            case DatasetRegistry.DSO1:
                from photoholmes.datasets.dso1 import DSO1Dataset

                return DSO1Dataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )
            case DatasetRegistry.DSO1_OSN:
                from photoholmes.datasets.osn import DSO1OSNDataset

                return DSO1OSNDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=True,
                )
            case DatasetRegistry.CASIA1_COPY_MOVE:
                from photoholmes.datasets.casia1 import Casia1CopyMoveDataset

                return Casia1CopyMoveDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )

            case DatasetRegistry.CASIA1_COPY_MOVE_OSN:
                from photoholmes.datasets.osn import Casia1CopyMoveOSNDataset

                return Casia1CopyMoveOSNDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=True,
                )
            case DatasetRegistry.CASIA1_SPLICING_OSN:
                from photoholmes.datasets.osn import Casia1SplicingOSNDataset

                return Casia1SplicingOSNDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=True,
                )
            case DatasetRegistry.CASIA1_SPLICING:
                from photoholmes.datasets.casia1 import Casia1SplicingDataset

                return Casia1SplicingDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )

            case DatasetRegistry.AUTOSPLICE100:
                from photoholmes.datasets.autosplice import Autosplice100Dataset

                return Autosplice100Dataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )
            case DatasetRegistry.AUTOSPLICE90:
                from photoholmes.datasets.autosplice import Autosplice90Dataset

                return Autosplice90Dataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )
            case DatasetRegistry.AUTOSPLICE75:
                from photoholmes.datasets.autosplice import Autosplice75Dataset

                return Autosplice75Dataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )

            case DatasetRegistry.TRACE_NOISE_EXO:
                from photoholmes.datasets.trace import TraceNoiseExoDataset

                return TraceNoiseExoDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )
            case DatasetRegistry.TRACE_NOISE_ENDO:
                from photoholmes.datasets.trace import TraceNoiseEndoDataset

                return TraceNoiseEndoDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )
            case DatasetRegistry.TRACE_CFA_ALG_EXO:
                from photoholmes.datasets.trace import TraceCFAAlgExoDataset

                return TraceCFAAlgExoDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )
            case DatasetRegistry.TRACE_CFA_ALG_ENDO:
                from photoholmes.datasets.trace import TraceCFAAlgEndoDataset

                return TraceCFAAlgEndoDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )
            case DatasetRegistry.TRACE_CFA_GRID_EXO:
                from photoholmes.datasets.trace import TraceCFAGridExoDataset

                return TraceCFAGridExoDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )
            case DatasetRegistry.TRACE_CFA_GRID_ENDO:
                from photoholmes.datasets.trace import TraceCFAGridEndoDataset

                return TraceCFAGridEndoDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )
            case DatasetRegistry.TRACE_JPEG_GRID_EXO:
                from photoholmes.datasets.trace import TraceJPEGGridExoDataset

                return TraceJPEGGridExoDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )
            case DatasetRegistry.TRACE_JPEG_GRID_ENDO:
                from photoholmes.datasets.trace import TraceJPEGGridEndoDataset

                return TraceJPEGGridEndoDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )
            case DatasetRegistry.TRACE_JPEG_QUALITY_EXO:
                from photoholmes.datasets.trace import TraceJPEGQualityExoDataset

                return TraceJPEGQualityExoDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )
            case DatasetRegistry.TRACE_JPEG_QUALITY_ENDO:
                from photoholmes.datasets.trace import TraceJPEGQualityEndoDataset

                return TraceJPEGQualityEndoDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )
            case DatasetRegistry.TRACE_HYBRID_EXO:
                from photoholmes.datasets.trace import TraceHybridExoDataset

                return TraceHybridExoDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )
            case DatasetRegistry.TRACE_HYBRID_ENDO:
                from photoholmes.datasets.trace import TraceHybridEndoDataset

                return TraceHybridEndoDataset(
                    dataset_path=dataset_path,
                    preprocessing_pipeline=preprocessing_pipeline,
                    load=load,
                    tampered_only=tampered_only,
                )

            case _:
                raise NotImplementedError(
                    f"Dataset '{dataset_name}' is not implemented."
                )
