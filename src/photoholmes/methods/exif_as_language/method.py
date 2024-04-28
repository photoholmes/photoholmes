# Code derived from https://github.com/hellomuffin/exif-as-language
import random
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from photoholmes.methods.base import BaseMethod, BenchmarkOutput
from photoholmes.methods.exif_as_language.clip import ClipModel
from photoholmes.methods.exif_as_language.config import (
    EXIFAsLanguageArchConfig,
    EXIFAsLanguageConfig,
    pretrained_arch,
)
from photoholmes.methods.exif_as_language.postprocessing import (
    exif_as_language_postprocessing,
)
from photoholmes.methods.exif_as_language.utils import (
    cosine_similarity,
    mean_shift,
    normalized_cut,
)
from photoholmes.utils.generic import load_yaml
from photoholmes.utils.patched_image import PatchedImage
from photoholmes.utils.pca import PCA


class EXIFAsLanguage(BaseMethod):
    """
    Implementation of Exif as Language method [Zheng et al., 2023].

    In this method the content of the image is contrasted with the exif information to
    detect any inconsistencies between what is "said" about the image and what the
    image is.
    For more details and instruction to download the weights, see the
    original implementation at:
        https://github.com/hellomuffin/exif-as-language

    Run the photoholmes CLI with the `adapt-weights` command to prune the weights
    to be used with this method.
    """

    def __init__(
        self,
        weights: Optional[Union[str, dict]] = None,
        arch_config: Union[
            EXIFAsLanguageArchConfig, Literal["pretrained"]
        ] = "pretrained",
        device: str = "cpu",
        seed: int = 44,
    ):
        """
        Args:
            weights (Optional[Union[str, dict]]): Path to the weights for the
                CLIP model. If None, the model will be initialized from scratch.
            arch_config (EXIFAsLanguageArchConfig | "pretrained"): The architecture
                configuration for the CLIP model. If "pretrained" is passed, the
                architecture from the paper will be used.
            device (str): Device to run the network. Default is "cpu".
            seed (int): Seed to be used in random operations. Default is 44.
        """
        if arch_config == "pretrained":
            arch_config = pretrained_arch

        random.seed(seed)
        super().__init__()

        clipNet = ClipModel(
            vision=arch_config.clip_model.vision,
            text=arch_config.clip_model.text,
            pooling=arch_config.clip_model.pooling,
        )

        if weights:
            checkpoint = torch.load(weights, map_location=device)  # type: ignore
            clipNet.load_state_dict(checkpoint)

        self.patch_size = arch_config.patch_size
        self.num_per_dim = arch_config.num_per_dim
        self.feat_batch_size = arch_config.feat_batch_size
        self.pred_batch_size = arch_config.pred_batch_size

        self.ms_window, self.ms_iter = arch_config.ms_window, arch_config.ms_iter
        self.net = clipNet
        self.net.eval()

        self.device = torch.device(device)

    def predict(
        self,
        image: Tensor,
    ) -> Tuple[NDArray, NDArray, float, NDArray, Tensor]:
        """
        Run ExifAsLanguage on an image. The image is expected to be in the range [0, 1].
        You can use the exif_as_language_preprocessing pipeline from
        photoholmes.methods.exif_as_language to preprocess the image.

        Args:
            image (Tensor): the preprocessed input image. [C, H, W], range: [0, 1].

        Returns:
            Tuple[NDArray, NDArray, float, NDArray, Tensor]:
                ms (np.ndarray): Consistency map, [H, W], range [0, 1].
                ncuts (np.ndarray): Localization map, [H, W], range [0, 1].
                score (float): Prediction score, higher indicates existence of
                    manipulation.
                out_pca (np.ndarray): PCA visualization, [H, W, 3].
                affinity_matrix (Tensor): Affinity matrix, [n_patches, n_patches].
        """
        # Initialize image and attributes
        height, width = image.shape[1:]
        p_img = self.init_img(image)
        # Precompute features for each patch
        with torch.no_grad():
            patch_features = self.get_patch_feats(
                p_img, batch_size=self.feat_batch_size
            )
        # PCA visualization
        pca = PCA(n_components=3, whiten=True)
        feature_transform = pca.fit_transform(patch_features.cpu().numpy())
        pred_pca_map = self.predict_pca_map(
            p_img, feature_transform, batch_size=self.pred_batch_size
        )

        # Predict consistency maps
        pred_maps = self.predict_consistency_maps(
            p_img, patch_features, batch_size=self.pred_batch_size
        ).numpy()

        # Produce a single response map
        ms = mean_shift(
            pred_maps.reshape((-1, pred_maps.shape[0] * pred_maps.shape[1])),
            pred_maps,
            window=self.ms_window,
            iter=self.ms_iter,
        )

        # Run clustering to get localization map
        ncuts = normalized_cut(pred_maps)
        # TODO: change resize to our own implementation
        out_ms = cv2.resize(ms, (width, height), interpolation=cv2.INTER_LINEAR)
        out_ncuts = cv2.resize(
            ncuts.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        out_pca = np.zeros((height, width, 3))
        p1, p3 = np.percentile(pred_pca_map, 0.5), np.percentile(pred_pca_map, 99.5)
        pred_pca_map = (pred_pca_map - p1) / (p3 - p1) * 255  # >0
        pred_pca_map[pred_pca_map < 0] = 0
        pred_pca_map[pred_pca_map > 255] = 255
        for i in range(3):
            out_pca[:, :, i] = cv2.resize(
                pred_pca_map[:, :, i], (width, height), interpolation=cv2.INTER_LINEAR
            )
        score = pred_maps.mean()
        affinity_matrix = self.generate_afinity_matrix(patch_features)

        return out_ms, out_ncuts, score, out_pca, affinity_matrix

    def benchmark(self, image: Tensor) -> BenchmarkOutput:
        """
        Benchmarks the Exif as language method using the provided image.

        Args:
            image (Tensor): the preprocessed input image. [C, H, W], range: [0, 1].

        Returns:
            BenchmarkOutput: Contains the heatmap, mask and
                detection.
        """
        ms, ncuts, _, _, _ = self.predict(image)

        return exif_as_language_postprocessing(
            {"heatmap": ms, "mask": ncuts, "detection": None}, self.device
        )

    def to_device(self, device: str):
        """Move method to device"""
        self.net.to(device)
        self.device = torch.device(device)

    def init_img(self, img: Tensor) -> PatchedImage:
        """
        Initialize the image to be used in the method. It will be divided into patches
        and preprocessed.

        Args:
            img (Tensor): The preprocessed input image. [C, H, W], range: [0, 1].

        Returns:
            PatchedImage: The image to be used in the method.
        """
        # Initialize image and attributes
        _, height, width = img.shape
        assert (
            min(height, width) > self.patch_size
        ), "Image must be bigger than patch size"
        img = img.to(self.device)
        p_img = PatchedImage(img, self.patch_size, num_per_dim=self.num_per_dim)

        return p_img

    def predict_consistency_maps(
        self, img: PatchedImage, patch_features: Tensor, batch_size: int = 64
    ):
        """
        Predict consistency maps for an image.

        Args:
            img (PatchedImage): The image to be used in the method.
            patch_features (Tensor): The features for each patch in the image.
            batch_size (int): Batch size to be used in the prediction. Defaults to 64.

        Returns:
            Tensor: The consistency maps for the image.
        """
        # For each patch, how many overlapping patches?
        spread = max(1, img.patch_size // img.stride)

        # Aggregate prediction maps; for each patch, compared to each other patch
        responses = torch.zeros(
            (
                img.max_h_idx + spread - 1,
                img.max_w_idx + spread - 1,
                img.max_h_idx + spread - 1,
                img.max_w_idx + spread - 1,
            )
        )
        # Number of predictions for each patch
        vote_counts = (
            torch.zeros(
                (
                    img.max_h_idx + spread - 1,
                    img.max_w_idx + spread - 1,
                    img.max_h_idx + spread - 1,
                    img.max_w_idx + spread - 1,
                )
            )
            + 1e-4
        )

        # Perform prediction
        for idxs in img.pred_idxs_gen(batch_size=batch_size):
            # a to be compared to b
            patch_a_idxs = idxs[:, :2]  # [B, 2]
            patch_b_idxs = idxs[:, 2:]  # [B, 2]

            # Convert 2D index into its 1D version
            a_idxs = np.ravel_multi_index(
                patch_a_idxs.T.tolist(), [img.max_h_idx, img.max_w_idx]
            )  # [B]
            b_idxs = np.ravel_multi_index(
                patch_b_idxs.T.tolist(), [img.max_h_idx, img.max_w_idx]
            )

            # Grab corresponding features
            a_feats = patch_features[a_idxs]  # [B, 4096]
            b_feats = patch_features[b_idxs]

            sim = self.patch_similarity(a_feats, b_feats)

            for i in range(len(sim)):
                responses[
                    idxs[i][0] : (idxs[i][0] + spread),
                    idxs[i][1] : (idxs[i][1] + spread),
                    idxs[i][2] : (idxs[i][2] + spread),
                    idxs[i][3] : (idxs[i][3] + spread),
                ] += sim[i]
                vote_counts[
                    idxs[i][0] : (idxs[i][0] + spread),
                    idxs[i][1] : (idxs[i][1] + spread),
                    idxs[i][2] : (idxs[i][2] + spread),
                    idxs[i][3] : (idxs[i][3] + spread),
                ] += 1

        # Normalize predictions
        return responses / vote_counts

    def predict_pca_map(
        self, img: PatchedImage, patch_features: NDArray, batch_size: int = 64
    ) -> NDArray:
        """
        Predict PCA visualization for an image.

        Args:
            img (PatchedImage): The image to be used in the method.
            patch_features (NDArray): The features for each patch in the image.
            batch_size (int): Batch size to be used in the prediction.
                Defaults to 64.

        Returns:
            NDArray: the PCA visualization for the image.
        """
        # For each patch, how many overlapping patches?
        spread = max(1, img.patch_size // img.stride)

        # Aggregate prediction maps; for each patch, compared to each other patch
        responses = np.zeros(
            (img.max_h_idx + spread - 1, img.max_w_idx + spread - 1, 3)
        )
        # Number of predictions for each patch
        vote_counts = (
            np.zeros((img.max_h_idx + spread - 1, img.max_w_idx + spread - 1, 3)) + 1e-4
        )

        # Perform prediction
        for idxs in img.idxs_gen(batch_size=batch_size):
            # a to be compared to b
            patch_a_idxs = idxs[:, :2]  # [B, 2]

            # Convert 2D index into its 1D version
            a_idxs = np.ravel_multi_index(
                patch_a_idxs.T.tolist(), [img.max_h_idx, img.max_w_idx]
            )  # [B]

            # Grab corresponding features
            a_feats = patch_features[a_idxs]  # [B, 3]

            for i in range(a_feats.shape[0]):
                responses[
                    idxs[i][0] : (idxs[i][0] + spread),
                    idxs[i][1] : (idxs[i][1] + spread),
                    :,
                ] += a_feats[i]
                vote_counts[
                    idxs[i][0] : (idxs[i][0] + spread),
                    idxs[i][1] : (idxs[i][1] + spread),
                    :,
                ] += 1

        # Normalize predictions
        return responses / vote_counts

    def patch_similarity(self, a_feats: Tensor, b_feats: Tensor) -> Tensor:
        """
        Compute similarity between two patches.

        Args:
            a_feats (Tensor): Features for patch a.
            b_feats (Tensor): Features for patch b.

        Returns:
            Tensor: Similarity between the two patches.
        """
        cos = cosine_similarity(a_feats, b_feats).diagonal()
        cos = 1 - cos
        cos = cos.cpu()
        return cos

    def get_patch_feats(self, img: PatchedImage, batch_size: int = 32):
        """
        Get features for every patch in the image. Features used to compute if two
        patches share the same EXIF attributes.

        Args:
            img (PatchedImage): The image to be used in the method.
            batch_size (int): Batch size to be fed into the network. Defaults to 32.

        Returns:
            Tensor: Features for each patch in the image.
        """
        # Compute feature vector for each image patch
        patch_features = []

        # Generator for patches; raster scan order
        for patches in img.patches_gen(batch_size):
            processed_patches = patches.to(self.device)
            feat = self.net.encode_image(processed_patches)

            if len(feat.shape) == 1:
                feat = feat.view(1, -1)
            patch_features.append(feat)

        # [n_patches, n_features]
        patch_features = torch.cat(patch_features, dim=0)

        return patch_features

    def generate_afinity_matrix(self, patch_features: Tensor) -> Tensor:
        """
        Generate affinity matrix for the patches in the image.

        Args:
            patch_features (Tensor): Features for each patch in the image.

        Returns:
            Tensor: Affinity matrix for the patches in the image.
        """
        patch_features = torch.nn.functional.normalize(patch_features)
        result = torch.matmul(patch_features, patch_features.t())

        return result

    def get_valid_patch_mask(self, mask: PatchedImage, batch_size: int = 32):
        """
        Get a mask for the valid patches in the image.

        Args:
            mask (PatchedImage): The mask to be used in the method.
            batch_size (int): Batch size to be fed into the network. Defaults to 32.

        Returns:
            Tensor: Mask for the valid patches in the image.
        """
        valid_mask = []
        for patches in mask.patches_gen(batch_size):
            patches = patches.reshape(
                patches.shape[0], -1
            )  # [batch_size, patch_size * patch_size]
            patches_sum = torch.sum(patches, dim=1)  # [batch_size]
            positive_mask = patches_sum > self.patch_size * self.patch_size * 0.9
            negative_mask = patches_sum == 0
            valid_mask.append(positive_mask.long() - negative_mask.long())
        valid_mask = torch.cat(valid_mask, dim=0)
        return valid_mask

    @classmethod
    def from_config(
        cls,
        config: Optional[EXIFAsLanguageConfig | dict | str | Path],
    ):
        """
        Instantiate the model from configuration dictionary or yaml.

        Params:
            config: Path to the yaml configuration or a dictionary with
                    the parameters for the model.
        """
        if isinstance(config, EXIFAsLanguageConfig):
            return cls(**config.__dict__)

        if isinstance(config, str) or isinstance(config, Path):
            config = load_yaml(str(config))
        elif config is None:
            config = {}

        exif_as_language_config = EXIFAsLanguageConfig(**config)

        return cls(
            arch_config=exif_as_language_config.arch_config,
            weights=exif_as_language_config.weights,
            device=exif_as_language_config.device,
            seed=exif_as_language_config.seed,
        )
