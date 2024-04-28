# code derived from https://www.grip.unina.it/download/prog/Splicebuster/
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.linalg import LinAlgWarning

from photoholmes.methods.base import BaseMethod, BenchmarkOutput
from photoholmes.utils.generic import load_yaml
from photoholmes.utils.pca import PCA

from .config import (
    FeaturesConfig,
    RegularImageFeaturesConfig,
    SaturationMaskConfig,
    SmallImageFeaturesConfig,
)
from .postprocessing import normalize_non_nan, resize_heatmap_and_pad
from .utils import (
    encode_matrix,
    feat_reduce_matrix,
    gaussian_mixture_mahalanobis,
    gaussian_uniform_mahalanobis,
    get_saturated_region_mask,
    quantize,
    third_order_residual,
)

logger = logging.getLogger(__name__)
YELLOW_COLOR = "\033[93m"
END_COLOR = "\033[0m"
warnings.filterwarnings("error", category=LinAlgWarning)


class Splicebuster(BaseMethod):
    """
    Implementation of the Splicebuster method [Cozzolino et al., 2015].

    This method is based on detecting splicing from features extracted from the image's
    residuals.
    """

    def __init__(
        self,
        image_size_threshold: int = 20000,
        small_image_feature_config: Union[
            FeaturesConfig, Literal["original"]
        ] = "original",
        regular_image_feature_config: Union[
            FeaturesConfig, Literal["original"]
        ] = "original",
        saturation_prob: float = 0.85,
        pca_dim: int = 25,
        pca: Literal["original", "uncentered", "correct"] = "original",
        mixture: Literal["uniform", "gaussian"] = "uniform",
        seed: Union[int, None] = 0,
        saturation_mask_config: Union[
            SaturationMaskConfig, Literal["original"], None
        ] = "original",
        **kwargs,
    ):
        """
        Initializes Splicebuster method class.

        Args:
            image_size_threshold (int): Threshold to determine if an image is small
                  or regular.
            small_image_config (FeaturesConfig | "original" | None): Feature
                  configuration for small images.
            regular_image_config (FeaturesConfig | "original" | None): Feature
                  configuration for regular sized images.
            pca_dim (int): Number of dimensions to keep after PCA. If 0, PCA is not
                  used.
            pca (str): PCA method to use. Options: 'original', 'uncentered', 'correct'.
                'original': PCA is applied to the features as in the original
                implementation.
                'uncentered': PCA is applied using sklearn but to the uncentered
                  features.
                'correct': PCA is applied using sklearn.
            mixture (str): Mixture model to use for mahalanobis distance estimation.
                Options: 'uniform', 'gaussian'.
            weight_params (WeightConfig | "original" | None): Provides parameters for
            weighted feature computation.
                None: Do not use weights.
                "original": Use parameters from the original implementation.
                WeightConfig object: Use custom parameters.
            seed (int | None): Random seed for mixture model initialization.
                default = 0.
        """
        super().__init__(**kwargs)

        logger.warning(
            f"{YELLOW_COLOR}Splicebuster is under a research only use license. "
            f"See the LICENSE inside the method folder.{END_COLOR}"
        )

        self.image_size_threshold = image_size_threshold
        self.small_image_feature_config = self._init_feature_config(
            small_image_feature_config, "small"
        )
        self.regular_image_feature_config = self._init_feature_config(
            regular_image_feature_config, "regular"
        )
        self.saturation_prob = saturation_prob
        self.pca_dim = pca_dim
        self.pca = pca

        self.weight_params = self._init_saturation_mask_config(saturation_mask_config)
        self.mahalanobis_estimation: Callable = self._init_mahal_estimation(mixture)
        self.seed = seed

    def _init_feature_config(
        self,
        feature_config: Union[FeaturesConfig, Literal["original"]],
        image_size: Optional[Literal["small", "regular"]],
    ) -> FeaturesConfig:
        """
        Initializes the feature configuration.

        Args:
            config (FeaturesConfig | "original" | None): Feature configuration.
            type (Optional[Literal["small", "regular"]]): Type of image to initialize.

        Returns:
            init_config (FeaturesConfig): Feature configuration.
        """
        if feature_config == "original":
            if image_size == "small":
                return SmallImageFeaturesConfig()
            elif image_size == "regular":
                return RegularImageFeaturesConfig()
            else:
                return FeaturesConfig()
        else:
            return feature_config

    def _init_saturation_mask_config(
        self,
        saturation_mask_config: Union[SaturationMaskConfig, Literal["original"], None],
    ) -> Optional[SaturationMaskConfig]:
        """
        Initializes the saturation mask configuration.

        Args:
            saturation_mask_config (SaturationMaskConfig | "original"): Saturation mask
                configuration.

        Returns:
            Optional[SaturationMaskConfig]: Saturation mask configuration.
        """
        if saturation_mask_config == "original":
            return SaturationMaskConfig()
        else:
            return saturation_mask_config

    def _init_mahal_estimation(
        self, mixture: Literal["uniform", "gaussian"]
    ) -> Callable:
        """
        Obtains the corresponding mahalanobis distance from a mixture model,
        according to the input 'mixture'.

        Args:
            mixture (str): String indicating the mixture model to use.
                Options: 'uniform', 'gaussian'.

        Returns:
            Function: Function to compute the mahalanobis distance.
        """
        if mixture == "gaussian":
            return gaussian_mixture_mahalanobis
        elif mixture == "uniform":
            return gaussian_uniform_mahalanobis
        else:
            raise ValueError(
                (
                    f"mixture {mixture} is not a valid mixture model. "
                    'Please select either "uniform" or "gaussian"'
                )
            )

    def _feature_config_case(self, image_size: Tuple[int, ...]) -> FeaturesConfig:
        """
        Gets the corresponding feature config, according to the image size,
        by looking at the product of the dimensions with respect to the attribute
        'image_size_threshold'.

        Arguments:
            image_size (Tuple[int, int]): Image size.

        Output:
            FeaturesConfig: corresponding feature config.
        """
        X, Y = image_size[:2]
        image_is_regular = X * Y > self.image_size_threshold
        return (
            self.regular_image_feature_config
            if image_is_regular
            else self.small_image_feature_config
        )

    def filter_and_encode(
        self, image: NDArray, q: int, T: int
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Apply third order residual filtering, quantization, and
        encode the result as base 3 integers for fast coocurrance counting.

        Args:
            image (np.ndarray): Image to process.
            q (int): Quantization parameter.
            T (int): Threshold for quantization.

        Returns:
            Tuple[NDArray, NDArray, NDArray, NDArray]: Tuple with the encoded
            residuals.
        """
        qh_res = quantize(third_order_residual(image), T, q)
        qv_res = quantize(third_order_residual(image, axis=1), T, q)

        qhh = encode_matrix(qh_res, T=T)
        qhv = encode_matrix(qh_res, T=T, axis=1)
        qvh = encode_matrix(qv_res, T=T)
        qvv = encode_matrix(qv_res, T=T, axis=1)

        return qhh, qhv, qvh, qvv

    def compute_histograms(
        self,
        qhh: NDArray[np.int64],
        qhv: NDArray[np.int64],
        qvh: NDArray[np.int64],
        qvv: NDArray[np.int64],
        stride: int,
        mask: Optional[NDArray] = None,
    ) -> Tuple[NDArray, NDArray, int, Tuple[NDArray, NDArray]]:
        """
        Efficiently compute histograms for stride x stride blocks.

        Args:
            qhh (np.ndarray): Encoded horizontal residuals.
            qhv (np.ndarray): Encoded horizontal-vertical residuals.
            qvh (np.ndarray): Encoded vertical-horizontal residuals.
            qvv (np.ndarray): Encoded vertical residuals.
            stride (int): Stride for the blocks.
            mask (np.ndarray | None): Mask to apply to the histograms. If None,
                no mask is applied.

        Returns:
            Tuple[NDArray, NDArray, int, Tuple[NDArray, NDArray]]:
                NDArray: Features.
                NDArray: Weights.
                int: Feature dimension.
                Tuple[NDArray, NDArray]: Coordinates.
        """
        H, W = qhh.shape
        x_range = np.arange(0, H - stride + 1, stride)
        y_range = np.arange(0, W - stride + 1, stride)

        if mask is None:
            mask = np.ones((H, W), dtype=np.uint8)

        n_bins = 1 + np.max((qhh, qhv, qvh, qvv))
        bins = np.arange(0, n_bins + 1)
        feat_dim = int(2 * n_bins)
        features = np.zeros((len(x_range), len(y_range), feat_dim))

        weights = np.zeros((len(x_range), len(y_range)))

        for x_i, i in enumerate(x_range):
            for x_j, j in enumerate(y_range):
                block_weights = mask[i : i + stride, j : j + stride]
                weights[x_i, x_j] = np.sum(block_weights)

                if weights[x_i, x_j] == 0:
                    continue

                Hhh = np.histogram(
                    qhh[i : i + stride, j : j + stride],
                    bins=bins,
                    weights=block_weights,
                    density=True,
                )[0].astype(float)
                Hvv = np.histogram(
                    qvv[i : i + stride, j : j + stride],
                    bins=bins,
                    weights=block_weights,
                    density=True,
                )[0].astype(float)
                Hhv = np.histogram(
                    qhv[i : i + stride, j : j + stride],
                    bins=bins,
                    weights=block_weights,
                    density=True,
                )[0].astype(float)
                Hvh = np.histogram(
                    qvh[i : i + stride, j : j + stride],
                    bins=bins,
                    weights=block_weights,
                    density=True,
                )[0].astype(float)

                features[x_i, x_j] = np.concatenate((Hhv + Hvh, Hhh + Hvv))

        weights /= stride**2

        return features, weights, feat_dim, (np.array(x_range), np.array(y_range))

    def correct_coords(
        self, coords: Tuple[NDArray, NDArray], block_size, stride
    ) -> Tuple[NDArray, NDArray]:
        """
        Apply correction to coordinates to account for the window filtering,
        coocurrance computation and center coordinate on window.

        Args:
            coords (Tuple[NDArray, NDArray]): Coordinates to correct.
            block_size (int): Size of the block.
            stride (int): Stride for the blocks.

        Returns:
            Tuple[NDArray, NDArray]: Corrected coordinates.
        """
        x_coords, y_coords = coords
        # window filtering
        x_coords += 4
        y_coords += 4
        # center coordinate on window
        x_coords = x_coords + (stride - 1) / 2
        y_coords = y_coords + (stride - 1) / 2

        # moving average compensation
        stride_x_block = block_size // stride
        low = int(np.floor((stride_x_block - 1) / 2))
        high = int(np.ceil((stride_x_block - 1) / 2))
        x_coords = (x_coords[low:-high] + x_coords[high:-low]) / 2
        y_coords = (y_coords[low:-high] + y_coords[high:-low]) / 2

        return x_coords, y_coords

    def compute_features(
        self, image: NDArray
    ) -> Tuple[NDArray, NDArray, Tuple[NDArray, NDArray]]:
        """
        Computes features, weights and coordinates for an image.

        Args:
            image (NDArray): Image to process.

        Returns:
            Tuple[NDArray, NDArray, Tuple[NDArray, NDArray]]:
                NDArray: Features.
                NDArray: Weights.
                Tuple[NDArray, NDArray]: Coordinates.
        """
        feature_config = self._feature_config_case(image.shape)
        block_size, stride, q, T = (
            feature_config.block_size,
            feature_config.stride,
            feature_config.q,
            feature_config.T,
        )
        qhh, qhv, qvh, qvv = self.filter_and_encode(image, q, T)

        if self.weight_params is not None:
            mask = get_saturated_region_mask(
                image,
                float(self.weight_params.low_th) / 255,
                float(self.weight_params.high_th) / 255,
            )

            mask = mask[4:-4, 4:-4]
            features, weights, feat_dim, coords = self.compute_histograms(
                qhh, qhv, qvh, qvv, stride, mask
            )
        else:
            features, weights, feat_dim, coords = self.compute_histograms(
                qhh, qhv, qvh, qvv, stride
            )

        strides_x_block = feature_config.block_size // feature_config.stride
        block_features = np.zeros(
            (
                features.shape[0] - strides_x_block + 1,
                features.shape[1] - strides_x_block + 1,
                feat_dim,
            )
        )
        block_weights = np.zeros(
            (
                features.shape[0] - strides_x_block + 1,
                features.shape[1] - strides_x_block + 1,
            )
        )
        for i in range(block_features.shape[0]):
            for j in range(block_features.shape[1]):
                block_weights[i, j] = weights[
                    i : i + strides_x_block, j : j + strides_x_block
                ].mean(axis=(0, 1))
                block_features[i, j] = features[
                    i : i + strides_x_block, j : j + strides_x_block
                ].mean(axis=(0, 1))

                block_features[i, j] /= np.maximum(block_weights[i, j], 1e-20)

        if self.pca_dim > 0:
            block_features = np.sqrt(block_features)

        coords = self.correct_coords(coords, block_size, stride)

        return block_features, block_weights, coords

    def _reduce_dimensions(
        self, flat_features: NDArray, valid_features: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        Reduces the dimensions of a set of features using PCA. The implementation used
        to calculate it varies according to attribute 'pca'.

        Args:
            flat_features (NDArray): Flattened features.
            valid_features (NDArray): Valid features.

        Returns:
            Tuple[NDArray, NDArray]: Tuple with the reduced flat and valid features.
        """
        if self.pca == "original":
            t = feat_reduce_matrix(self.pca_dim, valid_features)
            flat_features = np.matmul(flat_features, t)
            valid_features = np.matmul(valid_features, t)
        elif self.pca == "uncentered":
            pca = PCA(n_components=self.pca_dim, whiten=True)
            pca.fit(valid_features)
            # apply PCA over uncentered features as original implementation
            flat_features = pca.transform(flat_features + valid_features.mean(axis=0))
            valid_features = pca.transform(flat_features + valid_features.mean(axis=0))
        else:
            pca = PCA(self.pca_dim)
            valid_features = pca.fit_transform(valid_features)
            flat_features = pca.transform(flat_features)
        return flat_features, valid_features

    def predict(self, image: NDArray) -> NDArray:  # type: ignore[override]
        """
        Run splicebuster on an image.

        Args:
            image (NDArray): Grayscale image with dynamic range 0 and 1.

        Returns:
            NDArray: Splicebuster output
        """
        if image.ndim == 3:
            image = image[:, :, 0]
        X, Y = image.shape[:2]

        features, weights, coords = self.compute_features(image)
        valid = weights >= self.saturation_prob
        flat_features = features.reshape(-1, features.shape[-1])
        valid_features = flat_features[valid.flatten()]

        if len(valid_features) <= 1:
            return np.zeros((X, Y))

        if self.pca_dim > 0:
            flat_features, valid_features = self._reduce_dimensions(
                flat_features, valid_features
            )

        try:
            labels = self.mahalanobis_estimation(
                self.seed, valid_features, flat_features, valid
            )
        except LinAlgWarning:
            labels = np.zeros(flat_features.shape[0])
        heatmap = labels.reshape(features.shape[:2])
        heatmap = normalize_non_nan(heatmap)
        heatmap = resize_heatmap_and_pad(heatmap, coords, (X, Y))

        return heatmap

    def benchmark(self, image: NDArray) -> BenchmarkOutput:  # type: ignore[override]
        """
        Benchmarks the Splicebuster method using the provided image and size.

        Args:
            image (NDArray): Input image tensor.

        BenchmarkOutput:
            Contains the heatmap and placeholders for mask and detection.
        """
        heatmap = self.predict(image=image)

        return {
            "heatmap": torch.from_numpy(heatmap),
            "mask": None,
            "detection": None,
        }

    @classmethod
    def from_config(cls, config: Optional[str | Path | Dict[str, Any]]):
        """
        Instantiate the model from configuration dictionary or yaml.

        Params:
            config: path to the yaml configuration or a dictionary with
                    the parameters for the model.
        """
        if isinstance(config, (str, Path)):
            config = load_yaml(config)

        if config is None:
            config = {}

        if "saturation_mask_config" in config:
            config["saturation_mask_config"] = SaturationMaskConfig(
                **config["saturation_mask_config"]
            )
        if "regular_image_feature_config" in config:
            config["regular_image_feature_config"] = FeaturesConfig(
                **config["regular_image_feature_config"]
            )
        if "small_image_feature_config" in config:
            config["small_image_feature_config"] = FeaturesConfig(
                **config["small_image_feature_config"]
            )

        return cls(**config)
