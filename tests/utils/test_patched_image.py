import numpy as np
import pytest
import torch

from photoholmes.utils.patched_image import PatchedImage


class TestPatchedImage:
    STRIDE = 5
    PATCH_SIZE = 10

    @pytest.fixture
    def image(self):
        data_shape = (3, 100, 150)
        data = torch.rand(data_shape)
        patch_size = self.PATCH_SIZE
        stride = self.STRIDE
        return PatchedImage(data, patch_size=patch_size, stride=stride)

    def test_init(self, image: PatchedImage):
        assert image.stride == self.STRIDE
        assert image.patch_size == self.PATCH_SIZE
        assert image.max_h_idx == 19
        assert image.max_w_idx == 29

    def test_stride_calculation(self):
        data_shape = (3, 100, 100)
        data = torch.rand(data_shape)
        patch_size = 10
        num_per_dim = 5
        image = PatchedImage(data, patch_size, num_per_dim=num_per_dim)
        assert image.stride == 18

    def test_get_patch(self, image: PatchedImage):
        patch = image.get_patch(0, 0)
        assert patch.shape == (3, self.PATCH_SIZE, self.PATCH_SIZE)

    def test_get_patches(self, image: PatchedImage):
        idxs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        patches = image.get_patches(idxs)
        assert patches.shape == (idxs.shape[0], 3, self.PATCH_SIZE, self.PATCH_SIZE)

    def test_get_patch_map(self, image: PatchedImage):
        patch_map = image.get_patch_map(0, 0)
        assert patch_map.shape == image.shape[1:]
        assert patch_map.sum() == 10 * 10

    def test_patches_gen_count(self, image: PatchedImage):
        """Test that the patch generator yields the correct number of patches"""
        patches = list(image.patches_gen(1))
        assert len(patches) == image.max_h_idx * image.max_w_idx

    def test_patches_gen_batch_size(self, image: PatchedImage):
        """Test that the patch generator yields the correct number of patches"""
        patches = list(image.patches_gen(25))
        assert len(patches) == np.ceil(image.max_h_idx * image.max_w_idx / 25)
