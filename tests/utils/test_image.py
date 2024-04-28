from torch import Tensor

from photoholmes.utils.image import read_jpeg_data

JPEG_IMAGE_PATH = "tests/images/test_jpeg_image.jpeg"
PNG_IMAGE_PATH = "tests/images/png_test_image.png"


# --------- DCT channels LOAD ----------------------------------------------------------
def test_read_dct_channels_type():
    dct_channels, _ = read_jpeg_data(JPEG_IMAGE_PATH)
    assert isinstance(
        dct_channels, Tensor
    ), f"Expected Tensor, got {type(dct_channels)}"


def test_read_one_dct_channels():
    dct_channels, _ = read_jpeg_data(JPEG_IMAGE_PATH, num_dct_channels=1)
    assert (
        dct_channels.shape[0] == 1
    ), f"Expected 1 dct_channels, got {dct_channels.shape[0]}"


def test_read_two_dct_channels():
    dct_channels, _ = read_jpeg_data(JPEG_IMAGE_PATH, num_dct_channels=2)
    assert (
        dct_channels.shape[0] == 2
    ), f"Expected 3 dct_channels, got {dct_channels.shape[0]}"


def test_read_three_dct_channels():
    dct_channels, _ = read_jpeg_data(JPEG_IMAGE_PATH, num_dct_channels=3)
    assert (
        dct_channels.shape[0] == 3
    ), f"Expected 3 dct_channels, got {dct_channels.shape[0]}"


# --------- QUANT TABLE LOAD -----------------------------------------------------------
def test_read_quant_tables_type():
    _, qtables = read_jpeg_data(JPEG_IMAGE_PATH)
    assert isinstance(qtables, Tensor), f"Expected Tensor, got {type(qtables)}"


def test_read_one_quant_tables():
    _, qtables = read_jpeg_data(
        JPEG_IMAGE_PATH, num_dct_channels=1, all_quant_tables=False
    )
    assert qtables.ndim == 2, f"Expected 1 qtable, got {len(qtables)}"
    assert (
        qtables.shape[0] == 8 and qtables.shape[1] == 8
    ), f"Expected 8x8 qtable {qtables.shape}"


def test_read_all_quant_tables():
    _, qtables = read_jpeg_data(JPEG_IMAGE_PATH, all_quant_tables=True)
    assert qtables.ndim == 3, f"Expected a batch of qtables, got {len(qtables)}"
    assert (
        qtables.shape[1] == 8 and qtables.shape[2] == 8
    ), f"Expected 8x8 qtable, got {qtables.shape}"


# --------- READ PNG IMAGE WITH JPEG DATA ----------------------------------------------
def test_read_jpeg_data_with_png_image():
    dct_channels, qtables = read_jpeg_data(PNG_IMAGE_PATH)
    assert (
        dct_channels.shape[0] == 3
    ), f"Expected 1 dct_channels, got {dct_channels.shape[0]}"
    assert qtables.ndim == 2, f"Expected 1 qtable, got {qtables.shape[0]}"
