# Copyright contributors to the Terratorch project

import pytest
import torch

from terratorch.models.backbones.prithvi_vit import PRETRAINED_BANDS
from terratorch.tasks import ClassificationTask, PixelwiseRegressionTask, SemanticSegmentationTask

import gc

NUM_CHANNELS = 6
NUM_CLASSES = 2
EXPECTED_SEGMENTATION_OUTPUT_SHAPE = (1, NUM_CLASSES, 224, 224)
VIT_UPERNET_NECK = [
    {"name": "SelectIndices", "indices": [1, 2, 3, 4]},
    {"name": "ReshapeTokensToImage"},
    {"name": "LearnedInterpolateToPyramidal"},
]


@pytest.fixture(scope="session")
def model_factory() -> str:
    return "EncoderDecoderFactory"


@pytest.fixture(scope="session")
def model_input() -> torch.Tensor:
    return torch.ones((1, NUM_CHANNELS, 224, 224))


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100", "prithvi_eo_v2_300", "prithvi_swin_B"])
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder", "UNetDecoder"])
@pytest.mark.parametrize("loss", ["ce", "jaccard", "focal", "dice"])
@pytest.mark.parametrize("lr_overrides", [{"encoder": 0.01}, None])
def test_create_segmentation_task(backbone, decoder, loss, model_factory: str, lr_overrides):
    model_args = {
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "num_classes": NUM_CLASSES,
    }

    if decoder in ["UperNetDecoder", "UNetDecoder"] and backbone.startswith("prithvi_eo"):
        model_args["necks"] = VIT_UPERNET_NECK
    if decoder == "UNetDecoder":
        model_args["decoder_channels"] = [256, 128, 64, 32]
    SemanticSegmentationTask(
        model_args,
        model_factory,
        loss=loss,
        lr_overrides=lr_overrides,
    )

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100", "prithvi_eo_v2_300", "prithvi_swin_B"])
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder", "UNetDecoder"])
@pytest.mark.parametrize("loss", ["mae", "rmse", "huber"])
@pytest.mark.parametrize("lr_overrides", [{"encoder": 0.01}, None])
def test_create_regression_task(backbone, decoder, loss, model_factory: str, lr_overrides):
    model_args = {
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
    }

    if decoder in ["UperNetDecoder", "UNetDecoder"] and backbone.startswith("prithvi_eo"):
        model_args["necks"] = VIT_UPERNET_NECK
    if decoder == "UNetDecoder":
        model_args["decoder_channels"] = [256, 128, 64, 32]

    PixelwiseRegressionTask(
        model_args,
        model_factory,
        loss=loss,
        lr_overrides=lr_overrides,
    )

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100", "prithvi_eo_v2_300", "prithvi_swin_B"])
@pytest.mark.parametrize("decoder", ["FCNDecoder", "UperNetDecoder", "IdentityDecoder", "UNetDecoder"])
@pytest.mark.parametrize("loss", ["ce", "bce", "jaccard", "focal"])
@pytest.mark.parametrize("lr_overrides", [{"encoder": 0.01}, None])
def test_create_classification_task(backbone, decoder, loss, model_factory: str, lr_overrides):
    model_args = {
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": False,
        "num_classes": NUM_CLASSES,
    }

    if decoder in ["UperNetDecoder", "UNetDecoder"] and backbone.startswith("prithvi_eo"):
        model_args["necks"] = VIT_UPERNET_NECK
    if decoder == "UNetDecoder":
        model_args["decoder_channels"] = [256, 128, 64, 32]

    ClassificationTask(
        model_args,
        model_factory,
        loss=loss,
        lr_overrides=lr_overrides,
    )

    gc.collect()


@pytest.mark.parametrize("backbone", ["prithvi_eo_v1_100", "prithvi_eo_v2_300"])
@pytest.mark.parametrize("decoder", ["FCNDecoder"])
@pytest.mark.parametrize("vpt_n_tokens", [100, 500])
@pytest.mark.parametrize("vpt_dropout", [0.1, 0.5])
def test_create_task_with_vpt(
    backbone, decoder, vpt_n_tokens, vpt_dropout, model_factory: str, model_input
):
    model_args = {
        "backbone": backbone,
        "decoder": decoder,
        "backbone_bands": PRETRAINED_BANDS,
        "backbone_pretrained": True,
        "backbone_vpt": True,
        "backbone_vpt_n_tokens": vpt_n_tokens,
        "backbone_vpt_dropout": vpt_dropout,
        "num_classes": NUM_CLASSES,
    }

    task = SemanticSegmentationTask(
        model_args,
        model_factory,
        freeze_backbone=True,
    )


    with torch.no_grad():
        assert task.model(model_input).output.shape == EXPECTED_SEGMENTATION_OUTPUT_SHAPE
    
    n_layers = len(task.model.encoder.blocks)
    # count number of parameters
    n_params = sum(p.numel() for p in task.model.encoder.parameters() if p.requires_grad)
    assert n_params == (n_layers * vpt_n_tokens * task.model.encoder.embed_dim)

    gc.collect()
