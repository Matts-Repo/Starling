"""Native stacked load matches a concatenated reference (Section 10).

Replaces the external VDS concatenation script: DataSet._load_stacked_scans
assembles the multi-scan cube in memory from the per-scan entries, sorting by
the stack motor and building the motor meshgrid. This test exercises that
assembly with a fake reader and asserts it equals a hand-built concatenation.
"""

import os
import tempfile

import h5py
import numpy as np

import starling
from starling import DataSet


class _FakeReader:
    """Returns deterministic per-scan rocking blocks keyed by scan id."""

    def __init__(self, blocks, motor1):
        self.blocks = blocks
        self.motor1 = motor1
        self.scan_params = {
            "motor_names": ["mu"],
            "scan_shape": [len(motor1)],
            "integrated_motors": [False],
        }

    def __call__(self, scan_id, roi=None):
        return self.blocks[scan_id]


def test_stacked_load_matches_concatenation():
    rng = np.random.default_rng(0)
    a, b, m = 6, 5, 7  # detector 6x5, 7 rocking steps
    motor1 = np.linspace(4.7, 5.3, m).astype(np.float32)

    # three sub-scans stepped along obpitch, deliberately out of motor order
    scan_ids = ["1.1", "2.1", "3.1"]
    stack_vals = {"1.1": 0.30, "2.1": 0.10, "3.1": 0.20}  # unsorted on purpose
    blocks = {}
    for sid in scan_ids:
        block = rng.integers(0, 500, (a, b, m), dtype=np.uint16)
        motors = motor1[None, :].copy()
        blocks[sid] = (block, motors)

    # minimal h5 holding the stack-motor scalar inside each scan group
    tmp = tempfile.mkdtemp()
    h5path = os.path.join(tmp, "master.h5")
    with h5py.File(h5path, "w") as f:
        for sid in scan_ids:
            f.create_dataset(f"{sid}/obp", data=stack_vals[sid])

    ds = DataSet.__new__(DataSet)
    ds.h5file = h5path
    ds.reader = _FakeReader(blocks, motor1)
    ds.device = starling.get_device("cpu")
    ds.data = None
    ds.motors = None
    ds.roi = None

    ds._load_stacked_scans(list(scan_ids), "obp", roi=None, verbose=False)

    # expected: stacked along sorted motor order 2.1 (0.10), 3.1 (0.20), 1.1 (0.30)
    order = ["2.1", "3.1", "1.1"]
    ref = np.stack([blocks[s][0] for s in order], axis=-1)  # (a, b, m, 3)

    assert ds.data.shape == (a, b, m, 3)
    # same total frame count
    assert ds.data.shape[2] * ds.data.shape[3] == m * 3
    # identical detector spectra for every pixel
    assert np.array_equal(ds.data, ref)
    # motor grid: stack axis equals the sorted stack-motor values
    assert np.allclose(ds.motors[1, 0, :], [0.10, 0.20, 0.30])
    # rocking axis preserved
    assert np.allclose(ds.motors[0, :, 0], motor1)
