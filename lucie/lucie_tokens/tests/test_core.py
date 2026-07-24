import numpy as np

from lucie_tokens.data import parse_source_filename
from lucie_tokens.geometry import gaussian_latitude_weights
from lucie_tokens.schema import DEFAULT_SCALES, STATE_CHANNELS, scale_offsets


def test_static_contract():
    assert len(STATE_CHANNELS) == 33
    assert scale_offsets(DEFAULT_SCALES)[-1] == 388
    assert parse_source_filename("1981_0006.h5")[1] == 6


def test_latitude_weights_are_symmetric():
    latitude = np.linspace(-87.0, 87.0, 48)
    weights = gaussian_latitude_weights(latitude)
    np.testing.assert_allclose(weights, weights[::-1])
    assert np.isclose(weights.mean(), 1.0)

