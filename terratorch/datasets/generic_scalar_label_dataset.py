# Copyright contributors to the Terratorch project

"""Module containing generic dataset classes
"""
import glob
import os
from abc import ABC
from functools import partial
from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib as mpl
import numpy as np
import rioxarray
import torch
import xarray as xr
from albumentations.pytorch import ToTensorV2
from einops import rearrange
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torch import Tensor
from torchgeo.datasets import NonGeoDataset
from torchgeo.datasets.utils import rasterio_loader
from torchvision.datasets import ImageFolder

from terratorch.datasets.utils import HLSBands, filter_valid_files, to_tensor


class GenericScalarLabelDataset(NonGeoDataset, ImageFolder, ABC):
    """
    This is a generic dataset class to be used for instantiating datasets from arguments.
    Ideally, one would create a dataset class specific to a dataset.
    """

    def __init__(
        self,
        data_root: Path,
        split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        rgb_indices: list[int] | None = None,
        dataset_bands: list[HLSBands | int] | None = None,
        output_bands: list[HLSBands | int] | None = None,
        constant_scale: float = 1,
        transform: A.Compose | None = None,
        no_data_replace: float = 0,
        expand_temporal_dimension: bool = False,
    ) -> None:
        """Constructor

        Args:
            data_root (Path): Path to data root directory
            split (Path, optional): Path to file containing files to be used for this split.
                The file should be a new-line separated prefixes contained in the desired files.
                Files will be seached using glob with the form Path(data_root).glob(prefix + [image or label grep])
            ignore_split_file_extensions (bool, optional): Whether to disregard extensions when using the split
                file to determine which files to include in the dataset.
                E.g. necessary for Eurosat, since the split files specify ".jpg" but files are
                actually ".jpg". Defaults to True.
            allow_substring_split_file (bool, optional): Whether the split files contain substrings
                that must be present in file names to be included (as in mmsegmentation), or exact
                matches (e.g. eurosat). Defaults to True.
            rgb_indices (list[str], optional): Indices of RGB channels. Defaults to [0, 1, 2].
            dataset_bands (list[HLSBands | int] | None): Bands present in the dataset.
            output_bands (list[HLSBands | int] | None): Bands that should be output by the dataset.
            constant_scale (float): Factor to multiply image values by. Defaults to 1.
            transform (Albumentations.Compose | None): Albumentations transform to be applied.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            no_data_replace (float): Replace nan values in input images with this value. Defaults to 0.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Defaults to False.
        """
        self.split_file = split

        self.image_files = sorted(glob.glob(os.path.join(data_root, "**"), recursive=True))
        self.image_files = [f for f in self.image_files if not os.path.isdir(f)]
        self.constant_scale = constant_scale
        self.no_data_replace = no_data_replace
        self.expand_temporal_dimension = expand_temporal_dimension
        if self.expand_temporal_dimension and output_bands is None:
            msg = "Please provide output_bands when expand_temporal_dimension is True"
            raise Exception(msg)
        if self.split_file is not None:
            with open(self.split_file) as f:
                split = f.readlines()
            valid_files = {rf"{substring.strip()}" for substring in split}
            self.image_files = filter_valid_files(
                self.image_files,
                valid_files=valid_files,
                ignore_extensions=ignore_split_file_extensions,
                allow_substring=allow_substring_split_file,
            )

            def is_valid_file(x):
                return x in self.image_files

        else:

            def is_valid_file(x):
                return True

        super().__init__(
            root=data_root, transform=None, target_transform=None, loader=rasterio_loader, is_valid_file=is_valid_file
        )

        self.rgb_indices = [0, 1, 2] if rgb_indices is None else rgb_indices

        self.dataset_bands = self._generate_bands_intervals(dataset_bands)
        self.output_bands = self._generate_bands_intervals(output_bands)

        if self.output_bands and not self.dataset_bands:
            msg = "If output bands provided, dataset_bands must also be provided"
            return Exception(msg)  # noqa: PLE0101

        # There is a special condition if the bands are defined as simple strings.
        if self.output_bands:
            if len(set(self.output_bands) & set(self.dataset_bands)) != len(self.output_bands):
                msg = "Output bands must be a subset of dataset bands"
                raise Exception(msg)

            self.filter_indices = [self.dataset_bands.index(band) for band in self.output_bands]

        else:
            self.filter_indices = None
        # If no transform is given, apply only to transform to torch tensor
        self.transforms = transform if transform else lambda **batch: to_tensor(batch)
        # self.transform = transform if transform else ToTensorV2()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, label = ImageFolder.__getitem__(self, index)
        if self.expand_temporal_dimension:
            image = rearrange(image, "h w (channels time) -> time h w channels", channels=len(self.output_bands))
        if self.filter_indices:
            image = image[..., self.filter_indices]

        output = {
            "image": image.astype(np.float32) * self.constant_scale,
            "label": label,
            "filename": self.samples[index][
                0
            ],  # samples is an attribute of ImageFolder. Contains a tuple of (Path, Target)
        }
        if self.transforms:
            output = self.transforms(**output)
        return output

    def _load_file(self, path) -> xr.DataArray:
        data = rioxarray.open_rasterio(path, masked=True)
        data = data.fillna(self.no_data_replace)
        return data


class GenericNonGeoClassificationDataset(GenericScalarLabelDataset):
    """GenericNonGeoClassificationDataset"""

    def __init__(
        self,
        data_root: Path,
        num_classes: int,
        split: Path | None = None,
        ignore_split_file_extensions: bool = True,
        allow_substring_split_file: bool = True,
        rgb_indices: list[str] | None = None,
        dataset_bands: list[HLSBands | int] | None = None,
        output_bands: list[HLSBands | int] | None = None,
        class_names: list[str] | None = None,
        constant_scale: float = 1,
        transform: A.Compose | None = None,
        no_data_replace: float = 0,
        expand_temporal_dimension: bool = False,
    ) -> None:
        """Constructor

        Args:
            data_root (Path): Path to data root directory
            num_classes (int): Number of classes in the dataset
            split (Path, optional): Path to file containing files to be used for this split.
                The file should be a new-line separated prefixes contained in the desired files.
                Files will be seached using glob with the form Path(data_root).glob(prefix + [image or label grep])
            ignore_split_file_extensions (bool, optional): Whether to disregard extensions when using the split
                file to determine which files to include in the dataset.
                E.g. necessary for Eurosat, since the split files specify ".jpg" but files are
                actually ".jpg". Defaults to True.
            allow_substring_split_file (bool, optional): Whether the split files contain substrings
                that must be present in file names to be included (as in mmsegmentation), or exact
                matches (e.g. eurosat). Defaults to True.
            rgb_indices (list[str], optional): Indices of RGB channels. Defaults to [0, 1, 2].
            dataset_bands (list[HLSBands | int] | None): Bands present in the dataset.
            output_bands (list[HLSBands | int] | None): Bands that should be output by the dataset.
            class_names (list[str], optional): Class names. Defaults to None.
            constant_scale (float): Factor to multiply image values by. Defaults to 1.
            transform (Albumentations.Compose | None): Albumentations transform to be applied.
                Should end with ToTensorV2(). If used through the generic_data_module,
                should not include normalization. Not supported for multi-temporal data.
                Defaults to None, which simply applies ToTensorV2().
            no_data_replace (float): Replace nan values in input images with this value. Defaults to 0.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Defaults to False.
        """
        super().__init__(
            data_root,
            split=split,
            ignore_split_file_extensions=ignore_split_file_extensions,
            allow_substring_split_file=allow_substring_split_file,
            rgb_indices=rgb_indices,
            dataset_bands=dataset_bands,
            output_bands=output_bands,
            constant_scale=constant_scale,
            transform=transform,
            no_data_replace=no_data_replace,
            expand_temporal_dimension=expand_temporal_dimension,
        )
        self.num_classes = num_classes
        self.class_names = class_names

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        item["label"] = torch.tensor(item["label"]).long()
        return item

    def plot(self, sample: dict[str, Tensor], suptitle: str | None = None) -> Figure:
        pass


# TODO: Adapt to get the labels (Cant do it by getting them from the dir name as in classification)
# class GenericNonGeoRegressionDataset(GenericScalarLabelDataset):
#     """GenericNonGeoPixelwiseRegressionDataset"""

#     def __init__(
#         self,
#         data_root: Path,
#         split: Path | None = None,
#         ignore_split_file_extensions: bool = True,
#         allow_substring_split_file: bool = True,
#         rgb_indices: list[int] | None = None,
#         dataset_bands: list[HLSBands | int] | None = None,
#         output_bands: list[HLSBands | int] | None = None,
#         constant_scale: float = 1,
#         transform: A.Compose | None = None,
#         no_data_replace: float = 0,
#         expand_temporal_dimension: bool = False,
#     ) -> None:
#         """Constructor

#         Args:
#             data_root (Path): Path to data root directory
#             split (Path, optional): Path to file containing files to be used for this split.
#                 The file should be a new-line separated prefixes contained in the desired files.
#                 Files will be seached using glob with the form Path(data_root).glob(prefix + [image or label grep])
#             ignore_split_file_extensions (bool, optional): Whether to disregard extensions when using the split
#                 file to determine which files to include in the dataset.
#                 E.g. necessary for Eurosat, since the split files specify ".jpg" but files are
#                 actually ".jpg". Defaults to True.
#             allow_substring_split_file (bool, optional): Whether the split files contain substrings
#                 that must be present in file names to be included (as in mmsegmentation), or exact
#                 matches (e.g. eurosat). Defaults to True.
#             rgb_indices (list[str], optional): Indices of RGB channels. Defaults to [0, 1, 2].
#             dataset_bands (list[HLSBands | int] | None): Bands present in the dataset.
#             output_bands (list[HLSBands | int] | None): Bands that should be output by the dataset.
#             constant_scale (float): Factor to multiply image values by. Defaults to 1.
#             transform (Albumentations.Compose | None): Albumentations transform to be applied.
#                 Should end with ToTensorV2(). If used through the generic_data_module,
#                 should not include normalization. Not supported for multi-temporal data.
#                 Defaults to None, which simply applies ToTensorV2().
#             no_data_replace (float): Replace nan values in input images with this value. Defaults to 0.
#             expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
#                 Defaults to False.
#         """
#         super().__init__(
#             data_root,
#             split=split,
#             ignore_split_file_extensions=ignore_split_file_extensions,
#             allow_substring_split_file=allow_substring_split_file,
#             rgb_indices=rgb_indices,
#             dataset_bands=dataset_bands,
#             output_bands=output_bands,
#             constant_scale=constant_scale,
#             transform=transform,
#             no_data_replace=no_data_replace,
#             expand_temporal_dimension=expand_temporal_dimension,
#         )

#     def __getitem__(self, index: int) -> dict[str, Any]:
#         item = super().__getitem__(index)
#         item["label"] = torch.tensor(["label"]).float()
#         return item

#     def plot(self, sample: dict[str, Tensor], suptitle: str | None = None) -> Figure:
#         """Plot a sample from the dataset.

#         Args:
#             sample (dict[str, Tensor]): a sample returned by :meth:`__getitem__`
#             suptitle (str|None): optional string to use as a suptitle

#         Returns:
#             a matplotlib Figure with the rendered sample

#         .. versionadded:: 0.2
#         """
#         pass
