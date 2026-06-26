"""Raster/zigzag grid reconstruction in the 2-D reader (MosaScan).

The frames of an fscan2d arrive in acquisition order (slow outer, fast inner) and
a zigzag/snake scan reverses the fast sweep on alternate rows. The reader must
rebuild a monotonic, separable grid and — crucially — must do so even when the
slow motor's recorded values jitter within a row (encoder noise), which a plain
lexicographic ``(slow, fast)`` value sort scrambles.
"""

import warnings

import numpy as np
import pytest

from starling.io._reader import MosaScan, _warn_if_not_separable


def _acquisition_grid(m=4, n=5, zigzag=False, slow_jitter=0.0, seed=0):
    """Build a (1, 1, m, n) tagged stack + (2, m, n) motors in ACQUISITION order.

    data[0, 0, i, k] = slow_level*100 + fast_index of the cell acquired at row i,
    acquisition position k. Returns (data, motors, expected) where expected is the
    correctly-ordered tag grid (slow rows ascending, fast cols ascending).
    """
    rng = np.random.default_rng(seed)
    data = np.zeros((1, 1, m, n), dtype=np.int32)
    motors = np.zeros((2, m, n), dtype=np.float64)
    for i in range(m):
        s = 10.0 + i  # slow level value (step 1.0)
        for k in range(n):
            fast_idx = (n - 1 - k) if (zigzag and i % 2 == 1) else k
            data[0, 0, i, k] = i * 100 + fast_idx
            jit = rng.normal(0.0, slow_jitter) if slow_jitter else 0.0
            motors[0, i, k] = s + jit
            motors[1, i, k] = float(fast_idx)
    expected = np.array([[i * 100 + j for j in range(n)] for i in range(m)])
    return data, motors, expected


@pytest.mark.parametrize("zigzag", [False, True])
def test_snake_resort_reconstructs_grid(zigzag):
    data, motors, expected = _acquisition_grid(zigzag=zigzag)
    out, mot = MosaScan._resort_snake_by_acquisition(data, motors)
    assert np.array_equal(out[0, 0], expected)
    # slow constant along rows & ascending; fast ascending along cols & constant down
    assert np.all(np.diff(mot[0][:, 0]) > 0)
    assert np.allclose(mot[0], mot[0][:, :1])
    assert np.all(np.diff(mot[1][0]) > 0)
    assert np.allclose(mot[1], mot[1][:1, :])


def test_snake_resort_immune_to_slow_motor_jitter():
    # slow step is 1.0; inject 0.05 (5%) within-row jitter — enough that a
    # lexicographic value sort orders the fast axis by jitter instead.
    data, motors, expected = _acquisition_grid(zigzag=True, slow_jitter=0.05, seed=1)
    fixed, _ = MosaScan._resort_snake_by_acquisition(data.copy(), motors.copy())
    legacy, _ = MosaScan._resort_by_value(data.copy(), motors.copy())
    assert np.array_equal(fixed[0, 0], expected)          # acquisition-order: correct
    assert not np.array_equal(legacy[0, 0], expected)     # lexicographic: scrambled


def test_clean_data_matches_legacy():
    # with no jitter the new path must agree with the old value sort
    data, motors, expected = _acquisition_grid(zigzag=True, slow_jitter=0.0)
    new, _ = MosaScan._resort_snake_by_acquisition(data.copy(), motors.copy())
    legacy, _ = MosaScan._resort_by_value(data.copy(), motors.copy())
    assert np.array_equal(new[0, 0], legacy[0, 0])
    assert np.array_equal(new[0, 0], expected)


def test_separability_warns_on_scrambled_grid():
    # separable grid -> no warning
    _, motors, _ = _acquisition_grid()
    sep, _ = MosaScan._resort_snake_by_acquisition(
        *(_acquisition_grid()[0:2])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_not_separable(sep, "1.1")  # must not raise

    # slow motor varying wildly within rows -> warning
    bad = np.zeros((2, 3, 4))
    bad[0] = np.array([[0, 5, 1, 6], [10, 12, 11, 13], [20, 25, 21, 26]])  # not constant per row
    bad[1] = np.tile(np.arange(4), (3, 1))
    with pytest.warns(UserWarning, match="not cleanly separable"):
        _warn_if_not_separable(bad, "1.1")
