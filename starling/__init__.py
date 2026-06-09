import hdf5plugin  # noqa: F401  registers lima compression filters (bitshuffle/LZ4)

from . import device, preprocess, properties
from .device import get_device
from .io._dataset import DataSet

__version__ = "0.1.0"
__all__ = ["DataSet", "device", "get_device", "preprocess", "properties"]
