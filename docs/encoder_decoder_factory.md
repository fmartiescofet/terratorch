# EncoderDecoderFactory

## The Factory

A special factory provided by terratorch is the [EncoderDecoderFactory][terratorch.models.encoder_decoder_factory.EncoderDecoderFactory].

This factory leverages the `BACKBONE_REGISTRY`, `DECODER_REGISTRY` and `NECK_REGISTRY` to compose models formed as encoder + decoder, with some optional glue in between provided by the necks.
As most current models work this way, this is a particularly important factory, allowing for great flexibility in combining encoders and decoders from different sources.

The factory allows arguments to be passed to the encoder, decoder and head. Arguments with the prefix `backbone_` will be routed to the backbone constructor, with `decoder_` and `head_` working the same way. These are accepted dynamically and not checked.
Any unused arguments will raise a `ValueError`.

Both encoder and decoder may be passed as strings, in which case they will be looked in the respective registry, or as `nn.Modules`, in which case they will be used as is. In the second case, the factory assumes in good faith that the encoder or decoder which is passed conforms to the expected contract.

Not all decoders will readily accept the raw output of the given encoder. This is where necks come in. 
Necks are a sequence of operations which are applied to the output of the encoder before it is passed to the decoder.
They must be instances of [Neck][terratorch.models.necks.Neck], which is a subclass of `nn.Module`, meaning they can even define new trainable parameters.

The [EncoderDecoderFactory][terratorch.models.encoder_decoder_factory.EncoderDecoderFactory] returns a [PixelWiseModel][terratorch.models.pixel_wise_model.PixelWiseModel] or a [ScalarOutputModel][terratorch.models.scalar_output_model.ScalarOutputModel] depending on the task.

### :::terratorch.models.encoder_decoder_factory.EncoderDecoderFactory

### terratorch.models.pixel_wise_model.PixelWiseModel
### terratorch.models.scalar_output_model.ScalarOutputModel

## Encoders

To be a valid encoder, an object must be an `nn.Module` with an additional attribute `out_channels` which is a list of the channel dimension of the features it returns.

It's forward method should return a list of `torch.Tensor`.


## Necks

Necks are the glue between encoder and decoder. They can perform operations such as selecting elements from the output of the encoder ([SelectIndices][terratorch.models.necks.SelectIndices]), reshaping the outputs of ViTs so they are compatible with CNNs ([ReshapeTokensToImage][terratorch.models.necks.ReshapeTokensToImage]), amongst others.

Necks are `nn.Modules`, with an additional method `process_channel_list` which informs the [EncoderDecoderFactory][terratorch.models.encoder_decoder_factory.EncoderDecoderFactory] about how it will alter the channel list provided by `encoder.out_channels`.

### :::terratorch.models.necks.Neck

### :::terratorch.models.necks.SelectIndices

### :::terratorch.models.necks.PermuteDims

### :::terratorch.models.necks.InterpolateToPyramidal

### :::terratorch.models.necks.MaxpoolToPyramidal

### :::terratorch.models.necks.ReshapeTokensToImage

### :::terratorch.models.necks.AddBottleneckLayer

### :::terratorch.models.necks.LearnedInterpolateToPyramidal





## Decoders

To be a valid decoder, an object must be an `nn.Module` with an additional attribute `out_channels` which is an `int` with the channel dimension of the output.

The first argument to its constructor will be a list of channel dimensions it should expect as input.

It's forward method should accept a list of embeddings.

## Heads

Most decoders require a final head to be added for a specific task (e.g. semantic segmentation vs pixel wise regression).

Those registries producing decoders that dont require a head must expose the attribute `includes_head=True` so that a head is not added.
Decoders passed as `nn.Modules` which dont require a head must expose the same attribute themselves.

## :::terratorch.models.heads.classification_head.ClassificationHead

## :::terratorch.models.heads.regression_head.RegressionHead

## :::terratorch.models.heads.segmentation_head.SegmentationHead


## Decoder compatibilities

Not all encoders and decoders are compatible. Below we include some caveats.

Some decoders expect pyramidal outputs, but some encoders do not produce such outputs (e.g. vanilla ViT models).
In this case, the [InterpolateToPyramidal][terratorch.models.necks.InterpolateToPyramidal], [MaxpoolToPyramidal][terratorch.models.necks.MaxpoolToPyramidal] and [LearnedInterpolateToPyramidal][terratorch.models.necks.LearnedInterpolateToPyramidal] necks may be particularly useful.

### SMP decoders

Not all decoders are guaranteed to work with all encoders without additional necks.
Please check smp documentation to understand the embedding spatial dimensions expected by each decoder.

In particular, smp seems to assume the first feature in the passed feature list has the same spatial resolution
as the input, which may not always be true, and may break some decoders.

In addition, for some decoders, the final 2 features have the same spatial resolution.
Adding the [AddBottleneckLayer][terratorch.models.necks.AddBottleneckLayer] neck will make this compatible.

Some smp decoders require additional parameters, such as `decoder_channels`. These must be passed through the factory.
In the case of `decoder_channels`, it would be passed as `decoder_decoder_channels` (the first `decoder_` routes the parameter to the decoder, where it is passed as `decoder_channels`).

### MMSegmentation decoders

MMSegmentation decoders are available through the BACKBONE_REGISTRY. 

!!! warning

    MMSegmentation currently requires `mmcv==2.1.0`. Pre-built wheels for this only exist for `torch==2.1.0`.
    In order to use mmseg without building from source, you must downgrade your `torch` to this version.
    Install mmseg with:
    ``` sh
    pip install -U openmim
    mim install mmengine
    mim install mmcv==2.1.0
    pip install regex ftfy mmsegmentation
    ```

    We provide access to mmseg decoders as an external source of decoders, but are not directly responsible for the maintainence of that library.

Some mmseg decoders require the parameter `in_index`, which performs the same function as the `SelectIndices` neck.
For use for pixel wise regression, mmseg decoders should take `num_classes=1`.

