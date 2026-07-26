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

Aspect ratio (the sin(2theta) projection)
-----------------------------------------
The number above is the pixel size *in the detector plane*, and it is the
same in both directions -- the pco.edge sensor pixels are physically square
(6.5 x 6.5 um). It is NOT the pixel size in the sample.

ID03 has a **vertical scattering plane**: the condensed beam illuminates a
roughly horizontal sheet in the sample and the objective views that sheet
from 2theta above it (``obpitch`` IS 2theta -- the same angle appears as
``u = obx / cos(2theta)`` above, and a topotomo scan should return
``obpitch / 2 == mu``). A displacement ``dx`` ALONG THE BEAM in the sample
therefore lands on a detector that is perpendicular to the diffracted beam
as only ``dx * sin(2theta)``: the image is foreshortened along that one
direction, by 2.9x at a typical 2theta of 20 deg.

Undoing it means dividing the pixel size of that one axis by
``sin(2theta)``, exactly as darfix's Pixel Size widget does with its
"Aspect ratio: divide by sin(2θ) along [Vertical]" option (older darfix
called this the "Topography (obpitch)" checkbox)::

    d_vertical   = nominal / (M_crl * m_obj * sin(obpitch))
    d_horizontal = nominal / (M_crl * m_obj)

The foreshortened axis is the detector's VERTICAL, which is array axis 0
(image rows) for every starling cube -- see :data:`TOPOGRAPHY_AXIS`.
:func:`effective_pixel_size` returns the uncorrected detector-plane scalar;
:func:`sample_pixel_size` returns the ``(dy, dx)`` pair that maps a map onto
the sample and is what any figure with a physical axis or scale bar wants.
"""

import warnings

import numpy as np

NOMINAL_PIXEL_UM = 6.5
"""Physical pco.edge sensor pixel pitch in micrometres (unmagnified)."""

TOPOGRAPHY_AXIS = 0
"""Array axis carrying the sin(2theta)-foreshortened direction.

The raw detector dataset is ``(frame, y, x)`` and starling keeps that order,
so array axis 0 of every (ny, nx) map is the detector VERTICAL -- which, in
ID03's vertical scattering geometry, is the along-the-beam direction in the
sample and hence the compressed one. Pass ``topography_axis=1`` to
:func:`sample_pixel_size` if a setup ever puts the scattering plane
horizontal, or ``None`` to disable the correction entirely.
"""

_MIN_SIN_2THETA = 1e-6
"""Below this |sin(obpitch)| there is no usable diffraction geometry.

Guards both obpitch=0 and the round-off at multiples of 180 deg, and stops a
stray zero-ish motor readback turning into a 1e15x image stretch.
"""


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


def aspect_factor(invariant_motors):
    """``1 / sin(2theta)`` — the sample-plane stretch of the foreshortened axis.

    ``obpitch`` is 2theta (see the module docstring). A value of 2.88 at the
    typical ID03 2theta of 20.3 deg means the raw image is compressed by that
    factor along :data:`TOPOGRAPHY_AXIS` and a map plotted on square pixels is
    wrong by the same factor.

    Args:
        invariant_motors (dict): motor-name -> value mapping with ``obpitch``
            (degrees).

    Returns:
        float: ``1 / |sin(radians(obpitch))|``, or ``numpy.nan`` when
        ``obpitch`` is missing or is a multiple of 180 deg (no diffraction
        geometry, so no defined projection — note that ``sin(radians(180))``
        is 1.2e-16 rather than 0, hence the tolerance rather than ``> 0``).
    """
    try:
        obpitch_deg = float(np.asarray(invariant_motors["obpitch"]).squeeze())
    except (KeyError, TypeError, ValueError):
        return float("nan")
    s = abs(np.sin(np.radians(obpitch_deg)))
    if not np.isfinite(s) or s < _MIN_SIN_2THETA:
        return float("nan")
    return 1.0 / s


def sample_pixel_size(invariant_motors, nominal_pixel_um=NOMINAL_PIXEL_UM,
                      topography_axis=TOPOGRAPHY_AXIS):
    """Sample-plane pixel size ``(dy, dx)`` in micrometres, aspect-corrected.

    The detector-plane pixel from :func:`effective_pixel_size`, with the axis
    named by ``topography_axis`` divided by ``sin(2theta)`` to undo the
    projection of the illuminated sample sheet onto the tilted detector (see
    the module docstring). This is the pair that ``extent=`` / scale bars /
    grain-area numbers want; the scalar from :func:`effective_pixel_size` is
    correct only for the un-foreshortened axis.

    Args:
        invariant_motors (dict): see :func:`magnification`; also ``ffsel`` and
            ``obpitch``.
        nominal_pixel_um (float): unmagnified sensor pixel pitch.
        topography_axis (int or None): 0 (detector vertical, the ID03 default
            — see :data:`TOPOGRAPHY_AXIS`), 1 (detector horizontal), or None
            to skip the correction and return an isotropic pair.

    Returns:
        tuple: ``(dy_um, dx_um)`` for array axes (0, 1). Degrades to the
        isotropic pair with a warning when ``obpitch`` is unusable, so a
        missing motor costs you the aspect correction rather than a crash.
    """
    eff = effective_pixel_size(invariant_motors, nominal_pixel_um)
    if topography_axis is None:
        return (eff, eff)
    if topography_axis not in (0, 1):
        raise ValueError(
            f"topography_axis must be 0, 1 or None, got {topography_axis!r}"
        )
    stretch = aspect_factor(invariant_motors)
    if not np.isfinite(stretch):
        warnings.warn(
            "sample_pixel_size: could not resolve obpitch — returning an "
            "ISOTROPIC pixel size with NO sin(2theta) aspect correction. "
            "Maps will be compressed along the beam direction.",
            stacklevel=2,
        )
        return (eff, eff)
    d = [eff, eff]
    d[topography_axis] = eff * stretch
    return (d[0], d[1])
