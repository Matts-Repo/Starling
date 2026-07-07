"""Multiprocess shared-memory read engine for the ID03 readers.

libhdf5 serialises every API call behind one process-global lock, so threaded
h5py reads gain exactly nothing (measured 1.0x) — parallel decompression
requires separate *processes* (measured 3.5x at 4, 5.9x at 8 on warm data).
Each worker opens its own h5py handle and reads a disjoint range of
destination frames, placing them (transposed, re-sort folded in — the same
``_place_frames`` as the serial path) directly into a
``multiprocessing.shared_memory`` buffer that becomes the final array. Only
paths, index ranges and dtype strings cross the process boundary — never
h5py objects.

Start method: ``spawn`` on macOS/Windows (fork is unsafe under Apple
frameworks and torch threads), ``fork`` on Linux (no per-worker package
re-import — which matters when site-packages live on NFS). Override with
``STARLING_MP_START=spawn|fork|forkserver``. Workers never touch torch, so a
forked (possibly CUDA-initialised) parent context is left alone.

Shared-memory lifetime: the parent creates (and alone tracks) the segment;
workers attach *untracked* — Python <= 3.12 registers attached segments with
the resource tracker too (bpo-38119), which would corrupt the parent's
bookkeeping or let an exiting worker unlink a segment the parent still owns.
The parent unlinks the name as soon as the load completes (the mapping stays
valid; a crash before that point is cleaned up by the resource tracker) and
the mapping itself is closed by a ``weakref.finalize`` when the returned
array is garbage collected.
"""

import os
import sys
import weakref
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context, shared_memory

import hdf5plugin  # noqa: F401  lima bitshuffle/LZ4 filters, also in workers
import numpy as np


def _mp_context():
    method = os.environ.get("STARLING_MP_START")
    if not method:
        method = "spawn" if sys.platform in ("darwin", "win32") else "fork"
    return get_context(method)


def _attach_untracked(name):
    """Attach to an existing segment without resource-tracker registration:
    the worker only borrows the segment; the parent owns, tracks and unlinks
    it. Python 3.13+ has ``track=False``; on <= 3.12 registration is
    unconditional (bpo-38119), so it is no-op'ed for the attach call."""
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        from multiprocessing import resource_tracker

        orig_register = resource_tracker.register
        resource_tracker.register = lambda *a, **kw: None
        try:
            return shared_memory.SharedMemory(name=name)
        finally:
            resource_tracker.register = orig_register


def _fill(shm, path, scan_id, data_name, full_shape, dtype_str, roi, perm,
          sub_index, chunks):
    import h5py

    from ._reader import _place_frames

    arr = np.ndarray(full_shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
    view = arr if sub_index is None else arr[..., sub_index]
    out_flat = view.reshape(view.shape[0], view.shape[1], -1)
    if not np.may_share_memory(out_flat, arr):  # must be a view, never a copy
        raise ValueError("shared buffer is not reshapeable to (rows, cols, -1)")
    with h5py.File(path, "r") as h5f:
        dset = h5f[scan_id][data_name]
        for lo, hi in chunks:
            _place_frames(dset, out_flat, perm, roi, lo, hi)
    # arr/view/out_flat go out of scope here, releasing the buffer export so
    # the caller can close the mapping


def _worker(job):
    """Read one disjoint set of destination chunks into the shared buffer.

    ``job`` holds only picklable primitives: (path, scan_id, data_name,
    shm_name, full_shape, dtype_str, roi, perm, sub_index, chunks).
    """
    (path, scan_id, data_name, shm_name, full_shape, dtype_str, roi, perm,
     sub_index, chunks) = job
    shm = _attach_untracked(shm_name)
    try:
        _fill(shm, path, scan_id, data_name, full_shape, dtype_str, roi, perm,
              sub_index, chunks)
    finally:
        shm.close()
    return len(chunks)


def _close_shm(shm):
    try:
        shm.close()
    except Exception:
        pass


def _unlink_shm(shm):
    try:
        shm.unlink()
    except Exception:
        pass


def read_jobs_shm(shape, dtype, jobs, n_workers):
    """Run read jobs against one fresh shared-memory buffer of ``shape``.

    Each job is (path, scan_id, data_name, roi, perm, sub_index, chunks):
    ``sub_index`` selects a ``[..., k]`` slice of the buffer (a stacked-scan
    sub-scan) or None for the whole buffer. Jobs must cover disjoint
    destination regions.

    Returns the shared-memory-backed ndarray; raises on any worker failure
    (the caller falls back to the serial path). The segment name is unlinked
    before returning and the mapping is closed when the array is collected.
    """
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    shm = shared_memory.SharedMemory(create=True, size=max(1, nbytes))
    try:
        packed = [
            (path, scan_id, data_name, shm.name, tuple(shape), dtype.str, roi,
             perm, sub_index, chunks)
            for (path, scan_id, data_name, roi, perm, sub_index, chunks) in jobs
        ]
        with ProcessPoolExecutor(
            max_workers=max(1, min(n_workers, len(packed))),
            mp_context=_mp_context(),
        ) as pool:
            # list() drains the iterator so the first worker exception raises
            list(pool.map(_worker, packed))
    except BaseException:
        _close_shm(shm)
        _unlink_shm(shm)
        raise
    arr = np.ndarray(tuple(shape), dtype=dtype, buffer=shm.buf)
    _unlink_shm(shm)  # name released now; the mapping lives until arr does
    weakref.finalize(arr, _close_shm, shm)
    return arr


def _split_contiguous(chunks, n_parts):
    """Contiguous split of the chunk list — adjacent destination frames stay
    with one worker, so workers do not interleave writes within cache lines."""
    bounds = np.linspace(0, len(chunks), n_parts + 1).astype(int)
    return [
        chunks[bounds[i]:bounds[i + 1]]
        for i in range(n_parts)
        if bounds[i] < bounds[i + 1]
    ]


def read_scan_shm(path, scan_id, data_name, out_shape, dtype, roi, perm,
                  n_workers, row_len=None):
    """Parallel single-scan read into a shared-memory detector-first array.

    ``out_shape`` is (rows, cols, *scan_shape); workers read disjoint
    contiguous destination-frame ranges.
    """
    from ._reader import _dest_chunks, _frames_per_chunk

    rows, cols = out_shape[0], out_shape[1]
    n_frames = int(np.prod(out_shape[2:]))
    fpc = _frames_per_chunk(n_frames, rows * cols * np.dtype(dtype).itemsize,
                            row_len)
    chunks = _dest_chunks(n_frames, fpc)
    jobs = [
        (path, scan_id, data_name, roi, perm, None, part)
        for part in _split_contiguous(chunks, n_workers)
    ]
    return read_jobs_shm(out_shape, dtype, jobs, n_workers)
