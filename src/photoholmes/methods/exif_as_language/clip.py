from typing import Dict, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

try:
    from transformers.modeling_outputs import BaseModelOutput
except ImportError:
    raise ImportError(
        "`transformers` package not found, please run `pip install transformers`"
    )


def load_vision_model(model_name: Literal["resnet50"]) -> nn.Module:
    """
    Load a vision encoder model.

    Args:
        model_name (Literal["resnet50"]): name of the model to load.

    Returns:
        nn.Module: vision encoder model.
    """
    match model_name.lower():
        case "resnet50":
            from torchvision.models import resnet50

            model = resnet50()
        case _:
            raise NotImplementedError(f"Model name {model_name} is not implemented.")

    return model


def load_text_model(model_name: Literal["distilbert"]) -> nn.Module:
    """
    Load a text encoder model.

    Args:
        model_name (Literal["distilbert"]): name of the model to load.

    Returns:
        nn.Module: text encoder model.
    """
    match model_name.lower():
        case "distilbert":
            from transformers import DistilBertConfig, DistilBertModel

            bert_config = DistilBertConfig()
            model = DistilBertModel(bert_config)
        case _:
            raise NotImplementedError(f"Model name {model_name} is not implemented.")

    return model


class MeanPooler(nn.Module):
    """Mean pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: Tensor):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


class ClsPooler(nn.Module):
    """CLS token pooling"""

    def __init__(self):
        super().__init__()
        self.cls_token_position = 0

    def forward(self, x: BaseModelOutput, attention_mask: Tensor):
        return x.last_hidden_state[:, self.cls_token_position, :]


class ClipModel(nn.Module):
    def __init__(
        self,
        vision: Literal["resnet50"],
        text: Literal["distilbert"],
        pooling: Literal["cls", "mean"],
    ):
        """
        Simple clip model using HF transformers and torchvision models.

        Args:
            vision (Literal["resnet50"]): vision encoder model name.
            text (Literal["distilbert"]): text encoder model name.
            pooling (Literal["cls", "mean"]): pooling strategy to use.
        """
        super().__init__()
        self.visual = load_vision_model(vision)
        self.visual.fc = nn.Linear(2048, 768)
        self.transformer = load_text_model(text)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        match pooling:
            case "cls":
                self.pooler = ClsPooler()
            case "mean":
                self.pooler = MeanPooler()

    def encode_image(self, image: Tensor) -> Tensor:
        """
        Encode an image.

        Args:
            image (Tensor): image tensor.

        Returns:
            Tensor: image embeddings.
        """
        return self.visual(image)

    def encode_text(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Encode text.

        Args:
            inputs (Dict[str, Tensor]): input tensors.

        Returns:
            Tensor: text embeddings.
        """
        out = self.transformer(**inputs)
        out = self.pooler(out, inputs["attention_mask"])
        return out

    def forward(
        self, image: Tensor, attention_mask: Tensor, input_ids: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Args:
            image (Tensor): image tensor.
            attention_mask (Tensor): attention mask tensor.
            input_ids (Tensor): input ids tensor.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: image embeddings, text embeddings, logit scale.
        """
        text = {"attention_mask": attention_mask, "input_ids": input_ids}
        image_embeds = self.encode_image(image)
        text_embeds = self.encode_text(text)
        return image_embeds, text_embeds, self.logit_scale.exp()
