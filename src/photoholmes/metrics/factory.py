from typing import List, Union

import numpy as np
from torchmetrics import MetricCollection

from photoholmes.metrics.registry import MetricRegistry


class MetricFactory:
    """
    MetricFactory class responsible for creating metric instances.

    Supported Metrics:
        - AUROC (Area Under the Receiver Operating Characteristic curve)
        - FPR (False Positive Rate)
        - IoU (Intersection over Union, also known as Jaccard Index)
        - MCC (Matthews Correlation Coefficient)
        - Precision
        - ROC (Receiver Operating Characteristic curve)
        - TPR (True Positive Rate, synonymous with Recall)
        - IoU_WEIGHTED_V1 (Weighted Intersection over Union, version 1)
        - F1_WEIGHTED_V1 (Weighted F1 Score, version 1)
        - MCC_WEIGHTED_V1 (Weighted Matthews Correlation Coefficient, version 1)
        - IoU_WEIGHTED_V2 (Weighted Intersection over Union, version 2)
        - F1_WEIGHTED_V2 (Weighted F1 Score, version 2)
        - MCC_WEIGHTED_V2 (Weighted Matthews Correlation Coefficient, version 2)
        - F1 (F1 Score)
        - mAUROC (mean Area Under the Receiver Operating Characteristic curve)

    Note:
     - The v1 metrics correspond to the mean versions of the metrics, while the v2
       correspond to the weighted versions of the metrics implemented by torchmetrics.
       Please refer to the documentation of the respective metrics for more details.

    Methods:
        load(metric_names: List[Union[str, MetricRegistry]]) -> List[Metric]:
            Instantiates and returns a list of metric objects corresponding to the
            specified metric names.
    """

    @staticmethod
    def load(metric_names: Union[List[str], List[MetricRegistry]]) -> MetricCollection:
        """
        Instantiates and returns a list of metric objects corresponding to the specified
        metric names.

        Args:
            metric_names (List[Union[str, MetricRegistry]]): A list of the names of the
                metrics to load.
                These can be strings representing the metric names or instances of the
                MetricRegistry enum.

        Returns:
            List[Metric]: A list of metric objects corresponding to the provided metric
                names.
                The order of the metric objects in the list will correspond to the
                order of names provided.

        Raises:
            ValueError: If the 'metric_names' list is empty, indicating that no metric
                names have been specified.
            NotImplementedError: If any of the metric names provided are not recognized
                or not implemented in the PhotoHolmes library.

        Examples:
            Loading a single metric:
            >>> metrics = MetricFactory.load(["auroc"])

            Loading multiple metrics:
            >>> metrics = MetricFactory.load(["auroc", MetricRegistry.PRECISION])

        """
        if not metric_names:
            raise ValueError("metric_names cannot be empty.")
        metrics = []
        for metric_name in metric_names:
            if isinstance(metric_name, str):
                metric_name = MetricRegistry(metric_name.lower())
            match metric_name:
                case MetricRegistry.AUROC:
                    from photoholmes.metrics import AUROC

                    metrics.append(AUROC(thresholds=list(np.linspace(0, 1, 100))))
                case MetricRegistry.mAUROC:
                    from photoholmes.metrics import mAuroc

                    metrics.append(mAuroc(thresholds=list(np.linspace(0, 1, 100))))
                case MetricRegistry.FPR:
                    from photoholmes.metrics import FPR

                    metrics.append(FPR())
                case MetricRegistry.IoU:
                    from photoholmes.metrics import IoU

                    metrics.append(IoU())
                case MetricRegistry.MCC:
                    from photoholmes.metrics import MCC

                    metrics.append(MCC())
                case MetricRegistry.Precision:
                    from photoholmes.metrics import Precision

                    metrics.append(Precision())
                case MetricRegistry.ROC:
                    from photoholmes.metrics import ROC

                    metrics.append(ROC(thresholds=list(np.linspace(0, 1, 100))))
                case MetricRegistry.TPR:
                    from photoholmes.metrics import TPR

                    metrics.append(TPR())
                case MetricRegistry.IoU_WEIGHTED_V1:
                    from photoholmes.metrics.IoU_weighted_v1 import IoU_weighted_v1

                    metrics.append(IoU_weighted_v1())
                case MetricRegistry.F1_WEIGHTED_V1:
                    from photoholmes.metrics.F1_weighted_v1 import F1_weighted_v1

                    metrics.append(F1_weighted_v1())
                case MetricRegistry.MCC_WEIGHTED_V1:
                    from photoholmes.metrics.MCC_weighted_v1 import MCC_weighted_v1

                    metrics.append(MCC_weighted_v1())
                case MetricRegistry.IoU_WEIGHTED_V2:
                    from photoholmes.metrics.IoU_weighted_v2 import IoU_weighted_v2

                    metrics.append(IoU_weighted_v2())
                case MetricRegistry.F1_WEIGHTED_V2:
                    from photoholmes.metrics.F1_weighted_v2 import F1_weighted_v2

                    metrics.append(F1_weighted_v2())
                case MetricRegistry.MCC_WEIGHTED_V2:
                    from photoholmes.metrics.MCC_weighted_v2 import MCC_weighted_v2

                    metrics.append(MCC_weighted_v2())
                case MetricRegistry.F1:
                    from photoholmes.metrics import F1Score

                    metrics.append(F1Score())
                case _:
                    raise NotImplementedError(
                        f"Metric '{metric_name}' is not implemented."
                    )

        return MetricCollection(metrics)
