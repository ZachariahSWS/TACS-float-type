"""Coastline overlay for 48x96 ERA5 T30 map plots.

Vectors are Natural Earth 1:110m coastlines (public domain,
naturalearthdata.com), shipped as ``assets/ne_110m_coastline.geojson``; 110m is
already finer than the 3.75 degree grid, so nothing more detailed is useful.

Grid convention of ``era5_T30_regridded.npz`` (verified 2026-10-14 against its
own orography field, which lines up with these coastlines to the cell): row 0 is
the SOUTHERNMOST Gauss-Legendre latitude, rows ascend northward, and column j is
centred on longitude ``j * 360 / nlon`` east of Greenwich. The Zenodo
``data_inp.npy`` orography is the same grid (correlation 1.0, no shift).

The LUCIE-3D Bridges source (per-frame h5, ``lat.npy`` ascending) is the same
grid too, so ``imshow(field, origin="lower")`` puts north up for both.
``draw_coastlines`` reads the image's ``origin`` and ``extent``, so the overlay
follows whatever the image does (a south-up image gets a south-up coastline).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

COASTLINE_PATH = Path(__file__).with_name("assets") / "ne_110m_coastline.geojson"


def gaussian_latitudes(nlat: int) -> np.ndarray:
    """Gauss-Legendre latitudes in degrees, ascending south to north."""
    nodes, _ = np.polynomial.legendre.leggauss(int(nlat))
    return np.degrees(np.arcsin(nodes))


@lru_cache(maxsize=1)
def _lonlat_lines() -> tuple[np.ndarray, ...]:
    features = json.loads(COASTLINE_PATH.read_text())["features"]
    lines = []
    for feature in features:
        geometry = feature["geometry"]
        parts = (
            geometry["coordinates"]
            if geometry["type"] == "MultiLineString"
            else [geometry["coordinates"]]
        )
        lines.extend(np.asarray(part, dtype=np.float64) for part in parts)
    return tuple(lines)


def coastline_pixels(nlat: int = 48, nlon: int = 96, roll: int = 0) -> list[np.ndarray]:
    """Coastlines as (N, 2) arrays of (column, row) pixel coordinates.

    ``roll`` matches a field shifted with ``np.roll(field, roll, axis=-1)``
    (e.g. ``roll=nlon // 2`` puts Greenwich in the middle of the map).

    Segments are split where they cross the grid's longitude seam (half a cell
    west of 0E), so no line is drawn across the whole map."""
    lats = gaussian_latitudes(nlat)
    dlon = 360.0 / nlon
    out = []
    for line in _lonlat_lines():
        # longitude -> [-dlon/2, 360 - dlon/2) so the seam sits on a cell edge
        lon = np.mod(line[:, 0] + roll * dlon + dlon / 2, 360.0) - dlon / 2
        col = lon / dlon
        # rows are not evenly spaced in latitude; interpolate on the Gauss
        # nodes and extrapolate linearly into the polar half-cells
        row = np.interp(line[:, 1], lats, np.arange(nlat), left=np.nan, right=np.nan)
        polar = np.isnan(row)
        if polar.any():
            south = line[:, 1] < lats[0]
            row[polar & south] = (line[polar & south, 1] - lats[0]) / (lats[1] - lats[0])
            row[polar & ~south] = (nlat - 1) + (line[polar & ~south, 1] - lats[-1]) / (
                lats[-1] - lats[-2]
            )
        breaks = np.where(np.abs(np.diff(col)) > nlon / 2)[0] + 1
        for c, r in zip(np.split(col, breaks), np.split(row, breaks)):
            if len(c) > 1:
                out.append(np.column_stack([c, r]))
    return out


def draw_coastlines(image, roll: int = 0, **line_kwargs) -> None:
    """Overlay coastlines on ``image``, an ``AxesImage`` from ``ax.imshow`` of a
    (nlat, nlon) field stored south-first.

    Honours the image's ``origin`` and ``extent`` and keeps the axis limits;
    ``roll`` as in ``coastline_pixels`` for a longitude-shifted field.
    Defaults to thin black lines; pass e.g. ``color="w"`` for dark colormaps."""
    nlat, nlon = image.get_array().shape[:2]
    left, right, bottom, top = image.get_extent()
    # row 0 sits at the bottom edge for origin="lower", at the top for "upper"
    y0, y1 = (bottom, top) if image.origin == "lower" else (top, bottom)
    style = {"color": "k", "linewidth": 0.6, "alpha": 0.8}
    style.update(line_kwargs)
    ax = image.axes
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for segment in coastline_pixels(nlat, nlon, roll):
        x = left + (segment[:, 0] + 0.5) / nlon * (right - left)
        y = y0 + (segment[:, 1] + 0.5) / nlat * (y1 - y0)
        ax.plot(x, y, **style)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
