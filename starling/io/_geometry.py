"""Effective detector pixel size from the ID03 imaging geometry.

The far-field camera images the sample through the objective (CRL) at a
geometric magnification M = d2 / d1 given by the thin-lens object/image
distances, both derived from motor positions (ID03 SOP; see e.g. the
MA-7029 beamline log)::

    d1 = obx / cos(obpitch)              object distance, sample -> objective
    d2 = |mainx| / cos(obpitch) - d1     image distance, objective -> detector

On top of the CRL magnification sits the detector-side visible-light
objective selected by ``ffsel``: ``ffsel == 0`` puts the 10x objective in
the beam path; any other value (e.g. -90) parks it and the magnification is
2x. The total is::

    effective pixel = nominal pixel / ((d2 / d1) * m_obj),  m_obj = 10 or 2

Reference values (MA-6278 magnification scan): 2x optics -> ~195 nm/px,
10x optics -> ~39 nm/px for the 6.5 um pco.edge sensor (their exact 5x
ratio is the 10x/2x objective swap). Cross-check: MA-6043 2024 data
(obx=258.9, mainx=-5000, obpitch=20.19, ffsel=0) -> d2/d1 = 18.3, 10x
objective -> 35.5 nm/px, matching the beamline-quoted ~35 nm.
"""

import warnings

import numpy as np

NOMINAL_PIXEL_UM = 6.5
"""Physical pco.edge sensor pixel pitch in micrometres (unmagnified)."""


def magnification(invariant_motors):
    """Geometric magnification M = d2/d1 from obx/mainx/obpitch.

    Args:
        invariant_motors (dict): motor-name -> value mapping, e.g.
            ``scan_params["invariant_motors"]``. Requires ``obx``, ``mainx``
            and ``obpitch`` (obpitch in degrees).

    Returns:
        float: the magnification, or ``numpy.nan`` when the motors are
        missing or the geometry is degenerate (d1 <= 0 or d2 <= 0).
    """
    try:
        obx = float(np.asarray(invariant_motors["obx"]).squeeze())
        mainx = float(np.asarray(invariant_motors["mainx"]).squeeze())
        obpitch_deg = float(np.asarray(invariant_motors["obpitch"]).squeeze())
    except (KeyError, TypeError, ValueError):
        return float("nan")
    cospitch = np.cos(np.radians(obpitch_deg))
    if cospitch == 0:
        return float("nan")
    d1 = obx / cospitch
    d2 = abs(mainx) / cospitch - d1
    if d1 <= 0 or d2 <= 0:
        return float("nan")
    return d2 / d1


def objective_magnification(invariant_motors, ffsel_tol=1.0):
    """Detector-side objective magnification from the ``ffsel`` motor.

    ``ffsel == 0`` (within ``ffsel_tol``) means the 10x visible-light
    objective sits in front of the detector; any other position (e.g. -90)
    parks it and the effective objective is 2x.

    Args:
        invariant_motors (dict): motor-name -> value mapping with ``ffsel``.
        ffsel_tol (float): |ffsel| below this counts as "in the beam".

    Returns:
        float: 10.0 or 2.0, or ``numpy.nan`` when ``ffsel`` is missing.
    """
    try:
        ffsel = float(np.asarray(invariant_motors["ffsel"]).squeeze())
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return 10.0 if abs(ffsel) < ffsel_tol else 2.0


def effective_pixel_size(invariant_motors, nominal_pixel_um=NOMINAL_PIXEL_UM):
    """Effective (sample-plane) pixel size in micrometres.

    ``nominal_pixel_um / (M_crl * m_obj)`` with the CRL magnification from
    :func:`magnification` and the 10x/2x detector objective from
    :func:`objective_magnification` (ffsel). Degrades gracefully with a
    warning: a missing ffsel assumes 2x (the parked default); an unresolvable
    obx/mainx/obpitch geometry falls back to ``nominal_pixel_um`` so
    downstream scale bars land in detector-space units rather than crashing.

    Args:
        invariant_motors (dict): see :func:`magnification`; also ``ffsel``.
        nominal_pixel_um (float): unmagnified sensor pixel pitch.

    Returns:
        float: effective pixel size in micrometres.
    """
    M = magnification(invariant_motors)
    if not np.isfinite(M) or M <= 0:
        warnings.warn(
            "effective_pixel_size: could not resolve obx/mainx/obpitch "
            f"geometry — falling back to the nominal {nominal_pixel_um} "
            "um/pixel (detector-space, NOT sample-space).",
            stacklevel=2,
        )
        return float(nominal_pixel_um)
    m_obj = objective_magnification(invariant_motors)
    if not np.isfinite(m_obj):
        warnings.warn(
            "effective_pixel_size: ffsel not found — assuming the 10x "
            "objective is OUT of the beam (2x). Pass ffsel in "
            "invariant_motors or override the pixel size if this is wrong.",
            stacklevel=2,
        )
        m_obj = 2.0
    return float(nominal_pixel_um) / (M * m_obj)
