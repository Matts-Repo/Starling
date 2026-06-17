"""Strain helpers: convert a peak-centre / COM map (degrees) to strain.

These replicate the two darfix conventions exactly so users stop hand-rolling
``cot(theta) * dtheta``. Inputs are motor angles in **degrees**; outputs are
dimensionless strain. Angles are converted to radians internally for the
trigonometry.

For absolute (rather than relative-to-reference) strain you need the
crystallography (unit cell, hkl, Bragg angle); that is intentionally kept out
of these helpers — see ``ccmth_to_strain`` in the module note.
"""

import numpy as np


def strain_from_ccmth(ccmth_com, ccmth_0=None):
    """Strain from a ccmth (theta-like) centre-of-mass map.

    ``eps = (ccmth - ccmth_0) / tan(ccmth_0)``  (angles in radians internally).

    Args:
        ccmth_com (numpy.ndarray): per-pixel ccmth peak/COM, in degrees.
        ccmth_0 (float): reference angle in degrees. Defaults to
            ``nanmedian(ccmth_com)``.

    Returns:
        numpy.ndarray: dimensionless strain, same shape as ``ccmth_com``; NaNs
        in the input propagate.
    """
    ccmth_com = np.asarray(ccmth_com, dtype=float)
    if ccmth_0 is None:
        ccmth_0 = np.nanmedian(ccmth_com)
    th = np.deg2rad(ccmth_com)
    th0 = np.deg2rad(float(ccmth_0))
    return (th - th0) / np.tan(th0)


def strain_from_obpitch(obpitch_com, obpitch_0=None):
    """Strain from an obpitch centre-of-mass map.

    ``eps = -(obpitch - obpitch_0) / (2 * tan(obpitch_0 / 2))``
    (angles in radians internally).

    Args:
        obpitch_com (numpy.ndarray): per-pixel obpitch peak/COM, in degrees.
        obpitch_0 (float): reference angle in degrees. Defaults to
            ``nanmedian(obpitch_com)``.

    Returns:
        numpy.ndarray: dimensionless strain, same shape as ``obpitch_com``;
        NaNs in the input propagate.
    """
    obpitch_com = np.asarray(obpitch_com, dtype=float)
    if obpitch_0 is None:
        obpitch_0 = np.nanmedian(obpitch_com)
    ob = np.deg2rad(obpitch_com)
    ob0 = np.deg2rad(float(obpitch_0))
    return -(ob - ob0) / (2.0 * np.tan(ob0 / 2.0))
