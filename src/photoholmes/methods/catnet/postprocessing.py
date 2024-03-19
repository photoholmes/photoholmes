from photoholmes.postprocessing.resizing import (
    resize_heatmap_with_trim_and_pad,
    simple_upscale_heatmap,
)


def catnet_postprocessing(pred_authentic, pred_tempered, original_image_size):
    """
    Postprocessing for the CatNet method.

    Args:
        pred_authentic: Predicted heatmap for the authentic class.
        pred_tempered: Predicted heatmap for the tempered class.
        original_image_size: Size of the original image.

    Returns:
        The postprocessed heatmaps for the authentic and tempered classes.
    """
    pred_authentic = simple_upscale_heatmap(pred_authentic, 4)
    pred_tempered = simple_upscale_heatmap(pred_tempered, 4)

    pred_authentic = resize_heatmap_with_trim_and_pad(
        pred_authentic, original_image_size
    )
    pred_tempered = resize_heatmap_with_trim_and_pad(pred_tempered, original_image_size)

    return pred_authentic, pred_tempered
