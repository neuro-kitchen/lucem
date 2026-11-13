import numpy as np
import pytest

import lucem

hist = pytest.importorskip("lucem.hist", exc_type=ImportError)


def _blobs(size=96, centres=((24, 24), (24, 70), (70, 48)), radius=7):
    yy, xx = np.mgrid[:size, :size]
    out = np.zeros((size, size), dtype=np.float32)
    for cy, cx in centres:
        out[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2] = 1.0
    return out


def test_presets_deconvolve_back_to_their_stain():
    s = hist.stains.preset("H-DAB")
    od = np.array(s["stains"][0]["od"])
    # A pixel of hematoxylin only: I = 255 · 10^(−od)
    rgb = np.broadcast_to((255 * 10 ** (-0.5 * od)).round().astype(np.uint8), (4, 4, 3))
    h, dab, _ = hist.stains.deconvolve(rgb, s)
    assert h.mean() > 0.3 and abs(dab.mean()) < 0.1


def test_tissue_finds_the_bright_square():
    image = np.zeros((128, 128), dtype=np.float32)
    image[32:96, 32:96] = 100.0
    t = hist.tissue.detect(image, 1.0, brightfield=False, sigma_um=2.0, min_area_um2=10.0, min_hole_um2=1.0)
    assert t.mask.shape == (128, 128) and t.mask[64, 64] == 1 and t.mask[4, 4] == 0
    assert len(t.regions) == 1


def test_cells_found_measured_and_exported():
    nuclei = _blobs()
    found = hist.cells.detect(nuclei, {"DAPI": nuclei}, 0.5, background_radius_um=0.0, threshold=0.5, min_area_um2=5.0, cell_expansion_um=2.0)
    assert len(found) == 3
    assert found.nuclei.shape == nuclei.shape and found.nuclei.max() == 3
    table = hist.export.table(found.objects, "blobs", 0.5)
    assert len(table["Object ID"]) == 3
    assert hist.export.geojson(found.objects)["type"] == "FeatureCollection"


def test_cells_grow_from_given_nuclei():
    nuclei = _blobs()
    yy, xx = np.mgrid[:96, :96]
    labels = np.zeros(nuclei.shape, dtype=np.uint32)
    labels[(yy - 24) ** 2 + (xx - 24) ** 2 <= 49] = 5
    found = hist.cells.from_nuclei(labels, nuclei, {"DAPI": nuclei}, 0.5, cell_expansion_um=2.0)
    assert len(found) == 1


def test_steps_check_and_run_on_an_image():
    step = hist.steps.cells(nuclear=0, threshold=0.5, background_radius_um=0.0)
    assert hist.steps.check(step)["action"] == "cells"
    with pytest.raises(ValueError):
        hist.steps.check({"action": "no such step"})
    image = lucem.io.sample_volume(64, 1)
    result = hist.steps.run(image, [step])
    assert len(result.notes) == 1
