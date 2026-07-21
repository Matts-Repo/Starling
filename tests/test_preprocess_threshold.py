"""darfix-parity threshold_removal: in-place semantics + non-destructive twin."""

import numpy as np

from starling import preprocess


def _stack():
    rng = np.random.default_rng(7)
    return rng.integers(0, 200, size=(8, 9, 5), dtype=np.uint16)


def test_bottom_threshold_zeroes_below():
    data = _stack()
    ref = data.copy()
    out = preprocess.threshold_removal(data, bottom=50)
    assert out is data  # in place
    assert (data[ref < 50] == 0).all()
    assert (data[ref >= 50] == ref[ref >= 50]).all()


def test_top_threshold_zeroes_above():
    data = _stack()
    ref = data.copy()
    preprocess.threshold_removal(data, top=150)
    assert (data[ref > 150] == 0).all()
    assert (data[ref <= 150] == ref[ref <= 150]).all()


def test_both_bounds():
    data = _stack()
    ref = data.copy()
    preprocess.threshold_removal(data, bottom=50, top=150)
    keep = (ref >= 50) & (ref <= 150)
    assert (data[~keep] == 0).all()
    assert (data[keep] == ref[keep]).all()


def test_none_is_noop():
    data = _stack()
    ref = data.copy()
    preprocess.threshold_removal(data)
    assert (data == ref).all()


def test_thresholded_non_destructive():
    data = _stack()
    ref = data.copy()
    out = preprocess.thresholded(data, bottom=100)
    assert (data == ref).all()
    assert out is not data
    assert (out[ref < 100] == 0).all()


def test_darfix_parity():
    # darfix: data[data < bottom] = 0; data[data > top] = 0
    data = _stack()
    expected = data.copy()
    expected[expected < 30] = 0
    expected[expected > 180] = 0
    got = preprocess.thresholded(data, bottom=30, top=180)
    assert (got == expected).all()
