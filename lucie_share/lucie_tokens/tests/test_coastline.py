import numpy as np

from lucie_tokens.coastline import coastline_pixels, gaussian_latitudes


def test_gaussian_latitudes_ascend_south_to_north():
    lat = gaussian_latitudes(48)
    assert lat[0] < -87 and lat[-1] > 87 and np.all(np.diff(lat) > 0)


def test_coastline_pixels_are_on_the_south_first_grid():
    segments = coastline_pixels(48, 96)
    points = np.concatenate(segments)
    # columns stay inside the grid's longitude span (no seam-crossing lines)
    assert points[:, 0].min() >= -0.5 and points[:, 0].max() < 95.5
    assert max(np.abs(np.diff(s[:, 0])).max() for s in segments) < 48
    # Antarctica's coast is near row 0 (south), the Arctic's near row 47
    assert points[:, 1].min() < 6 and points[:, 1].max() > 44
    # Gibraltar (5.6W, 36N) maps to col ~94.5 and the row of +36 deg
    lat = gaussian_latitudes(48)
    near = points[np.hypot(points[:, 0] - (360 - 5.6) / 3.75, points[:, 1] - np.interp(36, lat, np.arange(48))) < 1]
    assert len(near) > 0
