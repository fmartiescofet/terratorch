import json
import gc
import matplotlib.pyplot as plt
import pytest
from utils import create_dummy_h5

from terratorch.datasets import MChesapeakeLandcoverNonGeo
from torchgeo.datasets import unbind_samples


@pytest.fixture
def dummy_chesapeake_data(tmp_path) -> str:
    base_dir = tmp_path / "chesapeake"
    base_dir.mkdir()
    data_dir = base_dir / "m-chesapeake"
    data_dir.mkdir()
    dummy_label_map = {
        "sample_train": [0, 1],
        "sample_valid": [0, 1],
        "sample_test": [0, 1]
    }
    with open(data_dir / "label_stats.json", "w") as f:
        json.dump(dummy_label_map, f)
    partition_file = data_dir / "default_partition.json"
    dummy_partitions = {
        "train": ["sample_train"],
        "valid": ["sample_valid"],
        "test": ["sample_test"]
    }
    with open(partition_file, "w") as f:
        json.dump(dummy_partitions, f)
    keys = list(MChesapeakeLandcoverNonGeo.all_band_names)
    dummy_shape = (256, 256)
    for sample_id in dummy_label_map.keys():
        file_path = data_dir / f"{sample_id}.hdf5"
        create_dummy_h5(str(file_path), keys, dummy_shape, label=0)
    return str(base_dir)

def test_chesapeake_datamodule(dummy_chesapeake_data):
    from terratorch.datamodules import MChesapeakeLandcoverNonGeoDataModule

    bands = MChesapeakeLandcoverNonGeo.all_band_names
    datamodule = MChesapeakeLandcoverNonGeoDataModule(
        data_root=dummy_chesapeake_data,
        bands=bands,
    )
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    train_batch = next(iter(train_loader))
    assert "image" in train_batch, "Missing key 'image' in train batch"
    assert "mask" in train_batch, "Missing key 'label' in train batch"
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()
    val_batch = next(iter(val_loader))
    assert "image" in val_batch, "Missing key 'image' in validation batch"
    assert "mask" in val_batch, "Missing key 'label' in validation batch"
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    test_batch = next(iter(test_loader))
    assert "image" in test_batch, "Missing key 'image' in test batch"
    assert "mask" in test_batch, "Missing key 'label' in test batch"
    datamodule.setup("predict")
    predict_loader = datamodule.predict_dataloader()
    predict_batch = next(iter(predict_loader))
    assert "image" in predict_batch, "Missing key 'image' in predict batch"
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()
    val_batch = next(iter(val_loader))
    sample = unbind_samples(val_batch)[0]
    datamodule.plot(sample)
    plt.close()
    gc.collect()
