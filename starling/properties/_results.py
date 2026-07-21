"""Named result objects for the analysis API.

Bare ``(ny, nx, 6)`` arrays force callers to remember column indices
(``params[..., 2]`` is the centre... or was it the width?). These dataclasses
give every quantity a name while keeping a ``.raw`` array for back-compat with
existing code and the golden tests.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ._maps import (
    FWHM_FACTOR,
    mosaicity as _mosaicity,
    orientation_map as _orientation_map,
    orientation_stamp as _orientation_stamp,
)


def _to_h5(path, kind, fields, attrs=None):
    """Write a flat dict of arrays + a ``kind`` tag to an HDF5 file."""
    import h5py

    with h5py.File(path, "w") as f:
        f.attrs["result_kind"] = kind
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
        for k, v in fields.items():
            if v is None:
                continue
            arr = np.asarray(v)
            if arr.ndim >= 2:
                f.create_dataset(k, data=arr, compression="gzip", shuffle=True)
            else:
                f.create_dataset(k, data=arr)


@dataclass
class Gauss1DResult:
    """Per-pixel 1-D Gaussian + linear-background fit.

    Fields (each (ny, nx)): amplitude ``A``, ``sigma``, peak centre ``mu``,
    background slope ``k``, background intercept ``m``, and ``success`` (1.0
    where the fit converged).
    """

    A: np.ndarray
    sigma: np.ndarray
    mu: np.ndarray
    k: np.ndarray
    m: np.ndarray
    success: np.ndarray

    @property
    def fwhm(self):
        """Full width at half maximum, ``2.3548 * sigma``."""
        return FWHM_FACTOR * self.sigma

    @property
    def raw(self):
        """Legacy ``(ny, nx, 6)`` array ``[A, sigma, mu, k, m, success]``."""
        return np.stack(
            [self.A, self.sigma, self.mu, self.k, self.m, self.success], axis=-1
        )

    @classmethod
    def from_raw(cls, arr):
        """Build from the legacy ``(ny, nx, 6)`` array."""
        arr = np.asarray(arr)
        return cls(*(arr[..., i] for i in range(6)))

    @classmethod
    def from_dict(cls, d):
        """Rebuild from a field dict (the ``fwhm`` property, if present, is ignored)."""
        return cls(
            A=np.asarray(d["A"]), sigma=np.asarray(d["sigma"]), mu=np.asarray(d["mu"]),
            k=np.asarray(d["k"]), m=np.asarray(d["m"]), success=np.asarray(d["success"]),
        )

    def to_dict(self):
        return {
            "A": self.A, "sigma": self.sigma, "mu": self.mu,
            "k": self.k, "m": self.m, "success": self.success, "fwhm": self.fwhm,
        }

    def to_h5(self, path):
        _to_h5(path, "gauss1d", self.to_dict())


@dataclass
class GaussNDResult:
    """Per-pixel N-D Gaussian + constant-background fit.

    Fields: amplitude ``A`` (ny, nx), centre ``mu`` (ny, nx, D), covariance
    ``cov`` (ny, nx, D, D) in motor units, background ``c`` (ny, nx) and
    ``success`` (ny, nx).
    """

    A: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    c: np.ndarray
    success: np.ndarray

    @property
    def D(self):
        return self.mu.shape[-1]

    @property
    def fwhm(self):
        """Per-axis FWHM, ``2.3548 * sqrt(diag(cov))`` -> (ny, nx, D)."""
        diag = np.einsum("...ii->...i", self.cov)
        return FWHM_FACTOR * np.sqrt(np.clip(diag, 0.0, None))

    def mosaicity(self, mode="scalar", axes=None):
        """Orientation spread from the fitted covariance (see properties.mosaicity)."""
        return _mosaicity(self.cov, mode=mode, axes=axes)

    def orientation(self, axes=(0, 1), norm="dynamic", as_rgb=False):
        """Mean-orientation map from the fitted centre (see properties.orientation_map)."""
        return _orientation_map(self.mu, axes=axes, norm=norm, as_rgb=as_rgb)

    def orientation_stamp(self, axes=(0, 1), mask=None, **kw):
        """darfix-style fixed-range colour stamp of the fitted centre.

        ``mask`` should be the grain/ok mask (e.g. ``success > 0.5``); see
        :func:`starling.properties.orientation_stamp`.
        """
        m = mask if mask is not None else (np.asarray(self.success) > 0.5)
        return _orientation_stamp(self.mu, axes=axes, mask=m, **kw)

    @property
    def raw(self):
        """Flat array ``[A, mu_0.., cov(upper-tri, row-major).., c, success]``.

        For D=2 this is exactly the legacy ``fit_2D_gaussian`` layout
        ``[A, mu0, mu1, cov00, cov01, cov11, c, success]``.
        """
        D = self.D
        cols = [self.A]
        for i in range(D):
            cols.append(self.mu[..., i])
        for r in range(D):
            for s in range(r, D):
                cols.append(self.cov[..., r, s])
        cols += [self.c, self.success]
        return np.stack(cols, axis=-1)

    @classmethod
    def from_dict(cls, d):
        """Rebuild from a field dict (``A``, ``mu``, ``cov``, ``c``, ``success``)."""
        return cls(
            A=np.asarray(d["A"]), mu=np.asarray(d["mu"]), cov=np.asarray(d["cov"]),
            c=np.asarray(d["c"]), success=np.asarray(d["success"]),
        )

    def to_dict(self):
        return {
            "A": self.A, "mu": self.mu, "cov": self.cov,
            "c": self.c, "success": self.success,
        }

    def to_h5(self, path):
        _to_h5(path, "gaussND", self.to_dict())


@dataclass
class GaussNDTwoResult:
    """Per-pixel two-component N-D Gaussian + shared constant-background fit.

    Peaks are sorted by fitted amplitude, **descending** (peak 1 = major
    component) — unlike the 1-D two-peak fit, an N-D scan has no natural axis
    to order by. The per-peak fields are populated only where the two-peak
    model was selected (``n_peaks == 2``) and are zero elsewhere, following
    starling's zero-degenerate-pixels convention, so nothing leaks into
    ``mosaicity()``/``orientation()``/``separation()`` maps.

    Fields: amplitudes ``A1``/``A2`` (ny, nx), centres ``mu1``/``mu2``
    (ny, nx, D) and covariances ``cov1``/``cov2`` (ny, nx, D, D) in motor
    units, shared background ``c`` (ny, nx), ``n_peaks`` (ny, nx) uint8
    (0 = no valid fit, 1 = single Gaussian preferred, 2 = two-peak selected),
    the BIC scores ``bic1``/``bic2`` (ny, nx; lower is better, 1e30 where a
    fit failed) and ``success`` (ny, nx) = 1.0 exactly where ``n_peaks == 2``.
    """

    A1: np.ndarray
    mu1: np.ndarray
    cov1: np.ndarray
    A2: np.ndarray
    mu2: np.ndarray
    cov2: np.ndarray
    c: np.ndarray
    n_peaks: np.ndarray
    bic1: np.ndarray
    bic2: np.ndarray
    success: np.ndarray

    @property
    def D(self):
        return self.mu1.shape[-1]

    def _peak(self, peak):
        if peak == 1:
            return self.A1, self.mu1, self.cov1
        if peak == 2:
            return self.A2, self.mu2, self.cov2
        raise ValueError(f"peak must be 1 or 2, got {peak!r}")

    def fwhm(self, peak=1):
        """Per-axis FWHM of one peak, ``2.3548 * sqrt(diag(cov))`` -> (ny, nx, D)."""
        _, _, cov = self._peak(peak)
        diag = np.einsum("...ii->...i", cov)
        return FWHM_FACTOR * np.sqrt(np.clip(diag, 0.0, None))

    def mosaicity(self, peak=1, mode="scalar", axes=None):
        """Orientation spread of one peak (see properties.mosaicity)."""
        _, _, cov = self._peak(peak)
        return _mosaicity(cov, mode=mode, axes=axes)

    def orientation(self, peak=1, axes=(0, 1), norm="dynamic", as_rgb=False):
        """Mean-orientation map of one peak (see properties.orientation_map)."""
        _, mu, _ = self._peak(peak)
        return _orientation_map(mu, axes=axes, norm=norm, as_rgb=as_rgb)

    def separation(self):
        """Peak-separation maps: per-axis delta-mu and Mahalanobis distance.

        Returns:
            tuple: ``dmu`` (ny, nx, D) = ``mu1 - mu2`` in motor units, and
            ``dist`` (ny, nx) — the Mahalanobis distance
            ``sqrt(dmu^T inv((cov1 + cov2) / 2) dmu)`` (a dimensionless
            "how many pooled sigmas apart" separation). Both are zero where
            ``n_peaks != 2``.
        """
        from ._linalg import masked_inv_spd

        dmu = self.mu1 - self.mu2
        ny, nx = dmu.shape[:2]
        D = dmu.shape[-1]
        pooled = 0.5 * (self.cov1 + self.cov2)
        pinv, ok = masked_inv_spd(pooled.reshape(-1, D, D).astype(np.float64))
        dv = dmu.reshape(-1, D)
        d2 = np.einsum("pi,pij,pj->p", dv, pinv, dv)
        dist = np.sqrt(np.clip(d2, 0.0, None))
        sel = (np.asarray(self.n_peaks).reshape(-1) == 2) & ok
        dist = np.where(sel, dist, 0.0).reshape(ny, nx)
        dmu = np.where(
            (np.asarray(self.n_peaks) == 2)[..., None], dmu, 0.0
        )
        return dmu, dist

    @classmethod
    def from_dict(cls, d):
        """Rebuild from a field dict (``n_peaks`` recast to uint8)."""
        return cls(
            A1=np.asarray(d["A1"]), mu1=np.asarray(d["mu1"]),
            cov1=np.asarray(d["cov1"]),
            A2=np.asarray(d["A2"]), mu2=np.asarray(d["mu2"]),
            cov2=np.asarray(d["cov2"]),
            c=np.asarray(d["c"]),
            n_peaks=np.asarray(d["n_peaks"]).astype(np.uint8),
            bic1=np.asarray(d["bic1"]), bic2=np.asarray(d["bic2"]),
            success=np.asarray(d["success"]),
        )

    def to_dict(self):
        return {
            "A1": self.A1, "mu1": self.mu1, "cov1": self.cov1,
            "A2": self.A2, "mu2": self.mu2, "cov2": self.cov2,
            "c": self.c, "n_peaks": self.n_peaks,
            "bic1": self.bic1, "bic2": self.bic2, "success": self.success,
        }

    def to_h5(self, path):
        _to_h5(path, "gaussND_two", self.to_dict())


@dataclass
class PseudoVoigtResult:
    """Per-pixel 1-D pseudo-Voigt + linear-background fit.

    Fields (each (ny, nx)): ``A``, Gaussian width ``sigma``, centre ``mu``,
    Lorentzian width ``gamma``, mixing ``eta`` in [0, 1] (1 = pure Lorentzian),
    background ``k``, ``m`` and ``success``.
    """

    A: np.ndarray
    sigma: np.ndarray
    mu: np.ndarray
    gamma: np.ndarray
    eta: np.ndarray
    k: np.ndarray
    m: np.ndarray
    success: np.ndarray

    @property
    def fwhm(self):
        """Approximate pseudo-Voigt FWHM from the Gaussian and Lorentzian widths.

        Uses the standard Thompson-Cox-Hastings width combination of the
        component FWHMs (fG = 2.3548 sigma, fL = 2 gamma).
        """
        fg = FWHM_FACTOR * self.sigma
        fl = 2.0 * self.gamma
        return (
            fg ** 5
            + 2.69269 * fg ** 4 * fl
            + 2.42843 * fg ** 3 * fl ** 2
            + 4.47163 * fg ** 2 * fl ** 3
            + 0.07842 * fg * fl ** 4
            + fl ** 5
        ) ** 0.2

    @property
    def raw(self):
        """Legacy ``(ny, nx, 8)`` array ``[A, sigma, mu, gamma, eta, k, m, success]``."""
        return np.stack(
            [self.A, self.sigma, self.mu, self.gamma, self.eta,
             self.k, self.m, self.success],
            axis=-1,
        )

    @classmethod
    def from_raw(cls, arr):
        arr = np.asarray(arr)
        return cls(*(arr[..., i] for i in range(8)))

    @classmethod
    def from_dict(cls, d):
        """Rebuild from a field dict (the ``fwhm`` property, if present, is ignored)."""
        return cls(
            A=np.asarray(d["A"]), sigma=np.asarray(d["sigma"]), mu=np.asarray(d["mu"]),
            gamma=np.asarray(d["gamma"]), eta=np.asarray(d["eta"]),
            k=np.asarray(d["k"]), m=np.asarray(d["m"]), success=np.asarray(d["success"]),
        )

    def to_dict(self):
        return {
            "A": self.A, "sigma": self.sigma, "mu": self.mu, "gamma": self.gamma,
            "eta": self.eta, "k": self.k, "m": self.m, "success": self.success,
        }

    def to_h5(self, path):
        _to_h5(path, "pseudovoigt", self.to_dict())


@dataclass
class MomentResult:
    """Per-pixel intensity-weighted moments.

    ``mean`` (ny, nx[, D]) and ``covariance`` (ny, nx[, D, D]) are always
    present; ``skew`` and ``kurtosis`` (per-axis, (ny, nx[, D])) are populated
    only when computed with ``order=4``.
    """

    mean: np.ndarray
    covariance: np.ndarray
    skew: Optional[np.ndarray] = None
    kurtosis: Optional[np.ndarray] = None

    def mosaicity(self, mode="scalar", axes=None):
        """Orientation spread from the moment covariance (see properties.mosaicity).

        Raw second moments are biased low by the finite motor window and
        residual background — prefer ``GaussNDResult.mosaicity`` for quantitative
        work.
        """
        return _mosaicity(self.covariance, mode=mode, axes=axes)

    def orientation(self, axes=(0, 1), norm="dynamic", as_rgb=False):
        """Mean-orientation map from the first moment (see properties.orientation_map)."""
        return _orientation_map(self.mean, axes=axes, norm=norm, as_rgb=as_rgb)

    def orientation_stamp(self, axes=(0, 1), mask=None, **kw):
        """darfix-style fixed-range colour stamp of the first moment.

        See :func:`starling.properties.orientation_stamp`.
        """
        return _orientation_stamp(self.mean, axes=axes, mask=mask, **kw)

    @classmethod
    def from_dict(cls, d):
        """Rebuild from a field dict; ``skew``/``kurtosis`` default to ``None``."""
        skew = d.get("skew")
        kurt = d.get("kurtosis")
        return cls(
            mean=np.asarray(d["mean"]), covariance=np.asarray(d["covariance"]),
            skew=None if skew is None else np.asarray(skew),
            kurtosis=None if kurt is None else np.asarray(kurt),
        )

    def to_dict(self):
        out = {"mean": self.mean, "covariance": self.covariance}
        if self.skew is not None:
            out["skew"] = self.skew
        if self.kurtosis is not None:
            out["kurtosis"] = self.kurtosis
        return out

    def to_h5(self, path):
        _to_h5(path, "moments", self.to_dict())
