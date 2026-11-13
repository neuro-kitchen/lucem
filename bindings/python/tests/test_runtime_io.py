import numpy as np

import lucem


def test_runtimes_list_cpu_and_it_is_in_use():
    assert "cpu" in lucem.runtime.available()
    assert lucem.runtime.current() == "cpu"


def test_sample_volume_reads_blocks_planes_and_regions():
    image = lucem.io.sample_volume(64, 8)
    assert image.channels and len(image.levels) >= 1
    depth, height, width = image.levels[0][:3]
    block = image.read_block(0, 0, [0, 0, 0], [2, 16, 16])
    assert block.shape == (2, 16, 16) and block.dtype == np.float32
    plane = image.read_plane(0, 0, "z", depth // 2)
    assert plane.shape == (height, width)
    region = image.read_region(0, 0, 8, 8, 16, 16, z=depth // 2)
    assert np.allclose(region, plane[8:24, 8:24])
    assert "Image(" in repr(image)


def test_the_catalog_gives_every_operator_a_halo():
    from lucem.ops import stage

    assert all("halo" in e for e in stage.CATALOG)
    assert stage.gaussian_blur(sigma=2.0).halo == 8
