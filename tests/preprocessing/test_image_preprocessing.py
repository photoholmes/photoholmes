import numpy as np
import pytest
import torch
from PIL import Image

from photoholmes.preprocessing.image import (
    Normalize,
    RGBtoGray,
    ToNumpy,
    ToTensor,
    ZeroOneRange,
)


class TestToTensor:
    """
    Tests for the ToTensor transform.
    """

    @pytest.fixture
    def to_tensor(self):
        return ToTensor()

    def test_to_tensor_3_channels(self, to_tensor: ToTensor):
        """
        Test that the ToTensor transform converts a three channel numpy image to a torch
        tensor.
        """
        np_image_shape = (100, 100, 3)
        dct_coeffs_shape = (100, 100, 2)

        np_image = np.random.rand(*np_image_shape).astype(np.float32)
        dct_coeffs = np.random.rand(*dct_coeffs_shape).astype(np.float32)

        result = to_tensor(np_image, dct_coefficients=dct_coeffs)

        assert isinstance(result, dict)
        assert isinstance(result["image"], torch.Tensor)
        assert isinstance(result["dct_coefficients"], torch.Tensor)
        assert result["image"].shape == (3, *np_image_shape[:2])
        assert result["dct_coefficients"].shape == dct_coeffs_shape

        np.testing.assert_allclose(
            result["image"].numpy(), np_image.transpose((2, 0, 1))
        )
        np.testing.assert_allclose(result["dct_coefficients"].numpy(), dct_coeffs)

    def test_to_tensor_1_channel(self, to_tensor: ToTensor):
        """
        Test that the ToTensor transform converts a one channel numpy image to a torch.
        """
        np_image_shape = (100, 100)
        np_image = np.random.rand(*np_image_shape).astype(np.float32)

        result = to_tensor(np_image)

        assert isinstance(result, dict)
        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].shape == np_image_shape

        np.testing.assert_allclose(result["image"].numpy(), np_image)


class TestToNumpy:
    @pytest.fixture
    def to_numpy(self):
        return ToNumpy()

    def test_to_numpy_tensor(self, to_numpy: ToNumpy):
        """
        Test that the ToNumpy transform converts a torch tensor image to a numpy array
        image.
        """
        # Create a torch tensor
        torch_image_shape = (3, 100, 100)
        torch_image = torch.rand(torch_image_shape)

        # Apply the ToNumpy transform
        result = to_numpy(image=torch_image)

        # Check that the output is a dictionary with a numpy array
        assert isinstance(result, dict)
        assert isinstance(result["image"], np.ndarray)
        assert result["image"].shape == (*torch_image_shape[1:], torch_image_shape[0])

        # Check that the values in the array are the same as in the tensor
        np.testing.assert_allclose(
            result["image"], torch_image.permute(1, 2, 0).numpy()
        )

    def test_to_numpy_pil(self, to_numpy: ToNumpy):
        """
        Test that the ToNumpy transform converts a PIL Image to a numpy array.
        """
        image_shape = (100, 100, 3)
        # Create a PIL Image
        pil_image = Image.fromarray(
            (np.random.rand(*image_shape) * 255).astype(np.uint8)
        )

        # Apply the ToNumpy transform
        result = to_numpy(image=pil_image)

        # Check that the output is a dictionary with a numpy array
        assert isinstance(result, dict)
        assert isinstance(result["image"], np.ndarray)
        assert result["image"].shape == (*image_shape[:2], image_shape[2])

        # Check that the values in the array are the same as in the PIL Image
        np.testing.assert_allclose(result["image"], np.array(pil_image))

    def test_to_numpy_kwargs_tensor(self, to_numpy: ToNumpy):
        """
        Test that the ToNumpy transform converts a torch tensor image to a numpy array.
        """
        # Create a torch tensor
        extra_shape = (6,)
        extra = torch.zeros(extra_shape)

        # Apply the ToNumpy transform
        result = to_numpy(extra=extra)

        # Check that the output is a dictionary with a numpy array
        assert isinstance(result, dict)
        assert isinstance(
            result["extra"], np.ndarray
        ), "Extra should come out as an array"
        assert result["extra"].shape == extra_shape
        assert np.allclose(result["extra"], extra.numpy())

    def test_to_numpy_kwargs_numpy(self, to_numpy: ToNumpy):
        """
        Test that the ToNumpy transform converts a numpy kwargs to a numpy array.
        """
        # Create a numpy array
        extra_shape = (6,)
        extra = np.zeros(extra_shape)

        # Apply the ToNumpy transform
        result = to_numpy(extra=extra)

        # Check that the output is a dictionary with a numpy array
        assert isinstance(result, dict)
        assert isinstance(
            result["extra"], np.ndarray
        ), "Extra should come out as an array"
        assert result["extra"].shape == extra_shape
        assert np.allclose(result["extra"], extra)

    def test_to_numpy_kwargs_other(self, to_numpy: ToNumpy):
        """
        Test that the ToNumpy transform converts a torch tensor kwarg to a numpy array.
        """
        # Create a list
        extra_len = 6
        extra = list(range(extra_len))

        # Apply the ToNumpy transform
        result = to_numpy(extra=extra)

        # Check that the output is a dictionary with a numpy array
        assert isinstance(result, dict)
        assert isinstance(
            result["extra"], np.ndarray
        ), "Extra should come out as an array"
        assert result["extra"].shape == (extra_len,)
        assert np.allclose(result["extra"], np.array(extra))


class TestRGBtoGray:
    @pytest.fixture
    def rgb_to_gray(self):
        return RGBtoGray()

    def test_rgb_to_gray_tensor(self, rgb_to_gray: RGBtoGray):
        # Create a torch tensor
        image_shape = (3, 100, 100)
        torch_image = torch.rand(image_shape)
        extra = "passthrough"

        # Apply the RGBtoGray transform
        result = rgb_to_gray(torch_image, extra=extra)

        # Check that the output is a dictionary with a torch tensor
        assert isinstance(result, dict)
        assert isinstance(result["image"], torch.Tensor)
        assert isinstance(
            result["extra"], str
        ), "Extra should be passed through unchanged"
        assert result["image"].shape == (1, *image_shape[1:])

    def test_rgb_to_gray_numpy(self, rgb_to_gray: RGBtoGray):
        # Create a numpy array
        image_shape = (100, 100, 3)
        np_image = np.random.rand(*image_shape).astype(np.float32)
        extra = "passthrough"

        # Apply the RGBtoGray transform
        result = rgb_to_gray(np_image, extra=extra)

        # Check that the output is a dictionary with a numpy array
        assert isinstance(result, dict)
        assert isinstance(result["image"], np.ndarray)
        assert isinstance(
            result["extra"], str
        ), "Extra should be passed through unchanged"
        assert result["image"].shape == (*image_shape[:2], 1)


class TestNormalize:
    @pytest.fixture
    def normalize(self):
        self.std = (0.5, 0.5, 0.5)
        self.mean = (0.5, 0.5, 0.5)
        return Normalize(mean=self.mean, std=self.std)

    def test_normalize_tensor(self, normalize: Normalize):
        # Create a torch tensor
        image_shape = (3, 100, 100)
        torch_image = torch.rand(image_shape)
        extra = "passthrough"

        # Apply the Normalize transform
        result = normalize(torch_image, extra=extra)

        # Check that the output is a dictionary with a torch tensor
        assert isinstance(result, dict)
        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].shape == image_shape
        assert isinstance(
            result["extra"], str
        ), "Extra should be passed through unchanged"

        # Check that the values in the array are in the range [-1, 1]
        assert result["image"].min() >= -1
        assert result["image"].max() <= 1

    def test_normalize_numpy(self, normalize: Normalize):
        # Create a numpy array
        image_shape = (100, 100, 3)
        np_image = np.random.rand(*image_shape).astype(np.float32)
        extra = "passthrough"

        # Apply the Normalize transform
        result = normalize(np_image, extra=extra)

        # Check that the output is a dictionary with a numpy array
        assert isinstance(result, dict)
        assert isinstance(result["image"], np.ndarray)
        assert result["image"].shape == image_shape
        assert isinstance(
            result["extra"], str
        ), "Extra should be passed through unchanged"

        # Check that the values in the array are in the range [-1, 1]
        assert result["image"].min() >= -1
        assert result["image"].max() <= 1

    def test_normalize_one_channel(self):
        normalize = Normalize(mean=(0.5,), std=(0.5,))
        image_shape = (100, 100)
        torch_image = torch.rand(image_shape)

        result = normalize(torch_image)

        assert isinstance(result, dict)
        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].shape == image_shape

        # Check that the values in the array are in the range [-1, 1]
        assert result["image"].min() >= -1
        assert result["image"].max() <= 1


class TestZeroOneRange:
    @pytest.fixture
    def zero_one_range(self):
        return ZeroOneRange()

    def test_zero_one_range_tensor(self, zero_one_range: ZeroOneRange):
        # Create a torch tensor
        image_shape = (3, 100, 100)
        torch_image = torch.rand(image_shape)
        extra = "passthrough"

        # Apply the ZeroOneRange transform
        result = zero_one_range(torch_image, extra=extra)

        # Check that the output is a dictionary with a torch tensor
        assert isinstance(result, dict)
        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].shape == image_shape
        assert isinstance(
            result["extra"], str
        ), "Extra should be passed through unchanged"

        # Check that the values in the array are in the range [0, 1]
        assert result["image"].min() >= 0
        assert result["image"].max() <= 1

    def test_zero_one_range_numpy(self, zero_one_range: ZeroOneRange):
        # Create a numpy array
        image_shape = (100, 100, 3)
        np_image = np.random.rand(*image_shape).astype(np.float32)
        extra = "passthrough"

        # Apply the ZeroOneRange transform
        result = zero_one_range(np_image, extra=extra)

        # Check that the output is a dictionary with a numpy array
        assert isinstance(result, dict)
        assert isinstance(result["image"], np.ndarray)
        assert result["image"].shape == image_shape
        assert isinstance(
            result["extra"], str
        ), "Extra should be passed through unchanged"

        # Check that the values in the array are in the range [0, 1]
        assert result["image"].min() >= 0
        assert result["image"].max() <= 1

    def test_zero_one_range_one_channel(self):
        zero_one_range = ZeroOneRange()
        image_shape = (100, 100)
        torch_image = torch.rand(image_shape)

        result = zero_one_range(torch_image)

        assert isinstance(result, dict)
        assert isinstance(result["image"], torch.Tensor)
        assert result["image"].shape == image_shape

        # Check that the values in the array are in the range [0, 1]
        assert result["image"].min() >= 0
        assert result["image"].max() <= 1
