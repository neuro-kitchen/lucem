import numpy as np
import pytest

from lucem import custom, io, ops
from lucem.ops import Pipeline, stage


def test_transform_with_halo_matches_the_whole_plane():
    image = io.sample_volume(96, 16)
    z = image.levels[0][0] // 2
    whole = image.read_block(0, 0, [z, 0, 0], [1, 96, 96])[0]
    k = 3

    def smooth(x):
        kernel = np.ones(2 * k + 1, dtype=np.float32) / (2 * k + 1)
        padded = np.pad(x, ((0, 0), (k, k)), mode="edge")
        return np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, padded)

    out = custom.transform(smooth, halo=k, block=32).apply(image)
    assert out.shape == (image.levels[0][0], 96, 96)
    assert np.allclose(out[z], smooth(whole), atol=1e-4)


def test_write_round_trips_through_ome_zarr(tmp_path):
    image = io.sample_volume(64, 12)
    path = custom.transform(lambda x: x * 2, block=32).write(image, tmp_path / "out.ome.zarr", channels=[1])
    back = io.open(path)
    assert back.channels == ["Filament"]
    assert np.allclose(back.read_block(0, 0, [0, 0, 0], [1, 64, 64]), image.read_block(0, 1, [0, 0, 0], [1, 64, 64]) * 2)


def test_measure_gives_a_row_per_block():
    image = io.sample_volume(64, 8)
    table = custom.measure(image, lambda x: {"mean": x.mean()}, channels=[0], block=32)
    assert len(table["mean"]) == image.levels[0][0] * 4
    assert set(table) == {"channel", "z", "y", "x", "mean"}


def test_a_pipeline_stays_on_the_device_and_reports_its_halo():
    image = io.sample_volume(64, 8)
    x = image.read_block(0, 0, [4, 0, 0], [1, 64, 64])[0]
    pipe = Pipeline([stage.gaussian_blur(sigma=1.0), stage.threshold(value=0.3), stage.open(radius=1)])
    assert pipe.halo == 4 + 2
    mask = pipe(x)
    assert mask.dtype == np.uint32
    assert set(np.unique(mask)) <= {0, 1}


def test_segment_joins_objects_across_blocks(tmp_path):
    image = io.sample_volume(96, 8)
    pipe = Pipeline([stage.threshold(value=0.5), stage.label()])
    z = image.levels[0][0] // 2
    table = custom.segment(image, pipe, tmp_path / "labels.ome.zarr", channel=1, block=24, halo=1)
    labels = io.open(tmp_path / "labels.ome.zarr").read_block(0, 0, [z, 0, 0], [1, 96, 96])[0]
    whole = Pipeline([stage.threshold(value=0.5), stage.label()])(image.read_block(0, 1, [z, 0, 0], [1, 96, 96])[0])
    assert len(np.unique(labels)) == len(np.unique(whole))
    assert table["area"].sum() > 0
    assert table["label"].max() == len(table["label"])


def test_operators_are_generated_from_the_catalog():
    x = np.zeros((16, 16), dtype=np.float32)
    assert ops.gaussian_blur(x, sigma=1.0).shape == (16, 16)
    with pytest.raises(TypeError):
        ops.gaussian_blur(x, sigmaa=1.0)
    a = np.array([[1, 0]], dtype=np.uint32)
    b = np.array([[1, 1]], dtype=np.uint32)
    assert ops.mask_subtract(b, a).tolist() == [[0, 1]]
