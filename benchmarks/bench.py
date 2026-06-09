"""Benchmark darling (numba CPU) vs starling (torch cpu/mps/cuda).

Usage: python benchmarks/bench.py [--full]
    --full uses a 2048x2048 detector (the real PCO frame size); default is
    600x600 (a typical grain auto-ROI).
"""

import argparse
import time

import numpy as np
import torch


def make_data(ny, nx, N=80, seed=1):
    rng = np.random.default_rng(seed)
    x = np.linspace(4.7385, 8.6385, N)
    A = rng.uniform(200, 5000, (ny, nx))
    mu = rng.uniform(x[5], x[-5], (ny, nx))
    sigma = rng.uniform(0.1, 0.5, (ny, nx))
    f = A[..., None] * np.exp(-0.5 * (x - mu[..., None]) ** 2 / sigma[..., None] ** 2) + 30
    return rng.poisson(np.clip(f, 0, None)).astype(np.uint16), x


def timeit(fn, repeat=3):
    fn()  # warmup (JIT / kernel compile)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="2048x2048 detector")
    args = ap.parse_args()

    ny = nx = 2048 if args.full else 600
    data, x = make_data(ny, nx)
    coords = np.array([x])
    mpix = ny * nx / 1e6
    print(f"detector {ny}x{nx} ({mpix:.2f} Mpx), {len(x)} motor points\n")

    import darling
    import starling

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    print("== moments ==")
    t = timeit(lambda: darling.properties.moments(data, coords))
    print(f"darling (numba)     {t:8.3f} s   {t / mpix:7.3f} s/Mpx")
    for dev in devices:
        t = timeit(lambda: starling.properties.moments(data, coords, device=dev))
        print(f"starling ({dev:4s})    {t:8.3f} s   {t / mpix:7.3f} s/Mpx")

    print("\n== fit_1D_gaussian ==")
    t_ref = timeit(lambda: darling.properties.curvefit.fit_1D_gaussian(data, (x,)), repeat=1)
    print(f"darling (numba)     {t_ref:8.3f} s   {t_ref / mpix:7.3f} s/Mpx")
    for dev in devices:
        t = timeit(
            lambda: starling.properties.fit_1D_gaussian(data, (x,), device=dev), repeat=1
        )
        print(
            f"starling ({dev:4s})    {t:8.3f} s   {t / mpix:7.3f} s/Mpx   "
            f"{t_ref / t:5.1f}x vs darling"
        )


if __name__ == "__main__":
    main()
