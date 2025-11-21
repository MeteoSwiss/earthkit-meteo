# (C) Copyright 2021 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
from typing import Literal
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import xarray as xr

from earthkit.utils.array import array_namespace
from numpy.typing import NDArray

from earthkit.meteo import constants

from support_operators import (
    TargetCoordinates,
    TargetCoordinatesAttrs,
    init_field_with_vcoord,
)

def pressure_at_model_levels(
    A: NDArray[Any], B: NDArray[Any], sp: Union[float, NDArray[Any]], alpha_top: str = "ifs"
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    r"""Compute pressure at model full- and half-levels.

    Parameters
    ----------
    A : ndarray
        A-coefficients defining the model levels. See [IFS-CY47R3-Dynamics]_
        (page 6) for details.
    B : ndarray
        B-coefficients defining the model levels. See [IFS-CY47R3-Dynamics]_
        (page 6) for details.
    sp : number or ndarray
        Surface pressure (Pa)
    alpha_top : str, optional
        Option to initialise alpha on the top of the model atmosphere (first half-level in vertical coordinate system). The possible values are:

        - "ifs": alpha is set to log(2). See [IFS-CY47R3-Dynamics]_ (page 7) for details.
        - "arpege": alpha is set to 1.0

    Returns
    -------
    ndarray
        Pressure at model full-levels
    ndarray
        Pressure at model half-levels
    ndarray
        Delta at full-levels
    ndarray
        Alpha at full levels


    Notes
    -----
    ``A`` and ``B`` must contain the same model half-levels in ascending order with
    respect to the model level number. The model level range must be contiguous and
    must include the bottom-most model half-level (surface), but not all the levels
    must be present. E.g. if the vertical coordinate system has 137 model levels using
    only a subset of levels between e.g. 137-96 is allowed.

    For details on the returned parameters see [IFS-CY47R3-Dynamics]_ (page 7-8).

    The pressure on the model-levels is calculated as:

    .. math::

        p_{k+1/2} = A_{k+1/2} + p_{s}\; B_{k+1/2}

        p_{k} = \frac{1}{2}\; (p_{k-1/2} + p_{k+1/2})

    where

        - :math:`p_{s}` is the surface pressure
        - :math:`p_{k+1/2}` is the pressure at the half-levels
        - :math:`p_{k}` is the pressure at the full-levels
        - :math:`A_{k+1/2}` and :math:`B_{k+1/2}` are the A- and B-coefficients defining
          the model levels.

    See also
    --------
    pressure_at_height_levels
    relative_geopotential_thickness

    """
    # constants
    PRESSURE_TOA = 0.1  # safety when highest pressure level = 0.0

    if alpha_top not in ["ifs", "arpege"]:
        raise ValueError(f"Unknown method '{alpha_top}' for pressure calculation. Use 'ifs' or 'arpege'.")

    alpha_top = np.log(2) if alpha_top == "ifs" else 1.0

    # make the calculation agnostic to the number of dimensions
    ndim = sp.ndim
    new_shape_half = (A.shape[0],) + (1,) * ndim
    A_reshaped = A.reshape(new_shape_half)
    B_reshaped = B.reshape(new_shape_half)

    # calculate pressure on model half-levels
    p_half_level = A_reshaped + B_reshaped * sp[np.newaxis, ...]

    # calculate delta
    new_shape_full = (A.shape[0] - 1,) + sp.shape
    delta = np.zeros(new_shape_full)
    delta[1:, ...] = np.log(p_half_level[2:, ...] / p_half_level[1:-1, ...])

    # pressure at highest half level<= 0.1
    if np.any(p_half_level[0, ...] <= PRESSURE_TOA):
        delta[0, ...] = np.log(p_half_level[1, ...] / PRESSURE_TOA)
    # pressure at highest half level > 0.1
    else:
        delta[0, ...] = np.log(p_half_level[1, ...] / p_half_level[0, ...])

    # calculate alpha
    alpha = np.zeros(new_shape_full)

    alpha[1:, ...] = (
        1.0 - p_half_level[1:-1, ...] / (p_half_level[2:, ...] - p_half_level[1:-1, ...]) * delta[1:, ...]
    )

    # pressure at highest half level <= 0.1
    if np.any(p_half_level[0, ...] <= PRESSURE_TOA):
        alpha[0, ...] = alpha_top
    # pressure at highest half level > 0.1
    else:
        alpha[0, ...] = (
            1.0 - p_half_level[0, ...] / (p_half_level[1, ...] - p_half_level[0, ...]) * delta[0, ...]
        )

    # calculate pressure on model full levels
    # TODO: is there a faster way to calculate the averages?
    # TODO: introduce option to calculate full levels in more complicated way
    p_full_level = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(2) / 2, mode="valid"), axis=0, arr=p_half_level
    )

    return p_full_level, p_half_level, delta, alpha


def relative_geopotential_thickness(
    alpha: NDArray[Any], delta: NDArray[Any], t: NDArray[Any], q: NDArray[Any]
) -> NDArray[Any]:
    """Calculate the geopotential thickness with respect to the surface on model full-levels.

    Parameters
    ----------
    alpha : array-like
        alpha term of pressure calculations
    delta : array-like
        delta term of pressure calculations
    t : array-like
        specific humidity on model full-levels (kg/kg).  First dimension must
        correspond to the model full-levels.
    q : array-like
        temperature on model full-levels (K).  First dimension must
        correspond to the model full-levels.

    Returns
    -------
    array-like
        geopotential thickness of model full-levels with respect to the surface

    Notes
    -----
    ``t`` and ``q`` must contain the same model levels in ascending order with respect to
    the model level number. The model level range must be contiguous and must include the
    bottom-most level, but not all the levels must be present. E.g. if the vertical coordinate
    system has 137 model levels using only a subset of levels between e.g. 137-96 is allowed.

    ``alpha`` and ``delta`` must be defined on the same levels as ``t`` and ``q``. These
    values can be calculated using :func:`pressure_at_model_levels`.

    The computations are described in [IFS-CY47R3-Dynamics]_ (page 7-8).

    See also
    --------
    pressure_at_model_levels

    """
    from earthkit.meteo.thermo import specific_gas_constant

    xp = array_namespace(alpha, delta, q, t)

    R = specific_gas_constant(q)
    d = R * t

    # compute geopotential thickness on half levels from 1 to NLEV-1
    dphi_half = xp.cumulative_sum(xp.flip(d[1:, ...] * delta[1:, ...], axis=0), axis=0)
    dphi_half = xp.flip(dphi_half, axis=0)

    # compute geopotential thickness on full levels
    dphi = xp.zeros_like(d)
    dphi[:-1, ...] = dphi_half + d[:-1, ...] * alpha[:-1, ...]
    dphi[-1, ...] = d[-1, ...] * alpha[-1, ...]

    return dphi


def pressure_at_height_levels(
    height: float,
    t: NDArray[Any],
    q: NDArray[Any],
    sp: NDArray[Any],
    A: NDArray[Any],
    B: NDArray[Any],
    alpha_top: str = "ifs",
) -> Union[float, NDArray[Any]]:
    """Calculate the pressure at a height above the surface from model full-levels.

    Parameters
    ----------
    height : number
        height above the surface for which the pressure needs to be computed (m)
    t : ndarray
        temperature at model full-levels (K). First dimension must
        correspond to the model full-levels.
    q : ndarray
        specific humidity at model full-levels (kg/kg). First dimension must
        correspond to the model full-levels.
    sp : ndarray
        surface pressure (Pa)
    A : ndarray
        A-coefficients defining the model levels
    B : ndarray
        B-coefficients defining the model levels
    alpha_top : str, optional
        Option passed to :func:`pressure_at_model_levels`. The possible values
        are: "ifs" (default) or "arpege".

    Returns
    -------
    number or ndarray
        pressure at the given height level (Pa)

    Notes
    -----
    ``t`` and ``q`` must contain the same model levels in ascending order with respect to
    the model level number. The model level range must be contiguous and must include the
    bottom-most level, but not all the levels must be present. E.g. if the vertical coordinate
    system has 137 model levels using only a subset of levels between e.g. 137-96 is allowed.

    ``A`` and ``B`` must be defined on the model half-levels corresponding to the model
    full-levels in ``t`` and ``q``. So the number of levels in ``A`` and ``B`` must be one
    more than the number of levels in ``t`` and ``q``.

    The pressure at height level is calculated by finding the model level above and
    below the specified height and interpolating the pressure with linear interpolation.

    See also
    --------
    pressure_at_model_levels
    relative_geopotential_thickness


    """
    # geopotential thickness of the height level
    tdphi = height * constants.g

    nlev = A.shape[0] - 1  # number of model full-levels

    # pressure(-related) variables
    p_full, p_half, delta, alpha = pressure_at_model_levels(A, B, sp, alpha_top=alpha_top)

    # relative geopotential thickness of full levels
    dphi = relative_geopotential_thickness(alpha, delta, t, q)

    # find the model full level right above the height level
    i_phi = (tdphi < dphi).sum(0)
    i_phi = i_phi - 1

    # TODO: handle case when height is above the highest model full-level
    # TODO: handle case when height is below the surface

    # initialize the output array
    p_height = np.zeros_like(i_phi, dtype=np.float64)

    # define mask: requested height is below the lowest model full-level
    mask = i_phi == nlev - 1

    # CASE 1: requested height is below the lowest model full-level
    # --> interpolation between surface pressure and lowest model full-level
    p_height[mask] = (p_half[-1, ...] + tdphi / dphi[-1, ...] * (p_full[-1, ...] - p_half[-1, ...]))[mask]

    # CASE 2: requested height is above the lowest model full-level
    # --> interpolation between between model full-level above and below

    # define some indices for masking and readability
    i_lev = i_phi
    indices = np.indices(i_lev.shape)
    masked_indices = tuple(dim[~mask] for dim in indices)
    above = (i_lev[~mask],) + masked_indices
    below = (i_lev[~mask] + 1,) + masked_indices

    dphi_above = dphi[above]
    dphi_below = dphi[below]

    # print(
    #     f"tdphi: {tdphi} above: {above} below: {below} dphi_above: {dphi_above} dphi_below  {dphi_below} p_full[above]: {p_full[above]} p_full[below]: {p_full[below]}"
    # )

    # calculate the interpolation factor
    factor = (tdphi - dphi_above) / (dphi_below - dphi_above)
    p_height[~mask] = p_full[above] + factor * (p_full[below] - p_full[above])

    return p_height


def geopotential_height_from_geopotential(z):
    r"""Compute geopotential height from geopotential.

    Parameters
    ----------
    z : array-like
        Geopotential (m2/s2)

    Returns
    -------
    array-like
        Geopotential height (m)


    The computation is based on the following definition:

    .. math::

        gh = \frac{z}{g}

    where :math:`g` is the gravitational acceleration on the surface of
    the Earth (see :py:attr:`meteo.constants.g`)
    """
    h = z / constants.g
    return h


def geopotential_from_geopotential_height(h):
    r"""Compute geopotential height from geopotential.

    Parameters
    ----------
    z : array-like
        Geopotential (m2/s2)

    Returns
    -------
    array-like
        Geopotential height (m)


    The computation is based on the following definition:

    .. math::

        z = gh\; g

    where :math:`g` is the gravitational acceleration on the surface of
    the Earth (see :py:attr:`meteo.constants.g`)
    """
    z = h * constants.g
    return z


def geopotential_height_from_geometric_height(h, R_earth=constants.R_earth):
    r"""Compute the geopotential height from geometric height.

    Parameters
    ----------
    h : array-like
        Geometric height with respect to the sea level (m)
    R_earth : float, optional
        Average radius of the Earth (m)

    Returns
    -------
    array-like
        Geopotential height (m)


    The computation is based on the following formula:

    .. math::

        gh = \frac{h\; R_{earth}}{R_{earth} + h}

    where :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`meteo.constants.R_earth`)
    """
    zh = h * R_earth / (R_earth + h)
    return zh


def geopotential_from_geometric_height(h, R_earth=constants.R_earth):
    r"""Compute the geopotential from geometric height.

    Parameters
    ----------
    h : array-like
        Geometric height with respect to the sea level (m)
    R_earth : float, optional
        Average radius of the Earth (m)

    Returns
    -------
    array-like
        Geopotential (m2/s2)


    The computation is based on the following formula:

    .. math::

        z = \frac{h\; g\; R_{earth}}{R_{earth} + h}

    where

        * :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`meteo.constants.R_earth`)
        * :math:`g` is the gravitational acceleration on the surface of
          the Earth (see :py:attr:`meteo.constants.g`)
    """
    z = h * R_earth * constants.g / (R_earth + h)
    return z


def geometric_height_from_geopotential_height(gh, R_earth=constants.R_earth):
    r"""Compute the geometric height from geopotential height.

    Parameters
    ----------
    gh : array-like
        Geopotential height (m)
    R_earth : float, optional
        Average radius of the Earth (m)

    Returns
    -------
    array-like
        Geometric height (m)


    The computation is based on the following formula:

    .. math::

        h = \frac{R_{earth}\; gh}{R_{earth} - gh}

    where :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`meteo.constants.R_earth`)
    """
    h = R_earth * gh / (R_earth - gh)
    return h


def geometric_height_from_geopotential(z, R_earth=constants.R_earth):
    r"""Compute the geometric height from geopotential.

    Parameters
    ----------
    z : array-like
        Geopotential (m2/s2)
    R_earth : float, optional
        Average radius of the Earth (m)

    Returns
    -------
    array-like
        Geometric height (m)


    The computation is based on the following formula:

    .. math::

        h = \frac{R_{earth} \frac{z}{g}}{R_{earth} - \frac{z}{g}}

    where

        * :math:`R_{earth}` is the average radius of the Earth (see :py:attr:`meteo.constants.R_earth`)
        * :math:`g` is the gravitational acceleration on the surface of
          the Earth (see :py:attr:`meteo.constants.g`)
    """
    z = z / constants.g
    h = R_earth * z / (R_earth - z)
    return h


def interpolate_k2p(
    field: xr.DataArray,
    mode: Literal["linear_in_p", "linear_in_lnp", "nearest_sfc"],
    p_field: xr.DataArray,
    p_tc_values: Sequence[float],
    p_tc_units: Literal["Pa", "hPa"],
) -> xr.DataArray:
    """Interpolate a field from model (k) levels to pressure coordinates.

    Example for vertical interpolation to isosurfaces of a target field,
    which is strictly monotonically decreasing with height.



    Parameters
    ----------
    field : xarray.DataArray
        field to interpolate (only typeOfLevel="generalVerticalLayer" is supported)
    mode : str
        interpolation algorithm, one of {"linear_in_p", "linear_in_lnp", "nearest_sfc"}
    p_field : xarray.DataArray
        pressure field on k levels in Pa
        (only typeOfLevel="generalVerticalLayer" is supported)
    p_tc_values : list of float
        pressure target coordinate values
    p_tc_units : str
        pressure target coordinate units

    Returns
    -------
    field_on_tc : xarray.DataArray
        field on target (i.e., pressure) coordinates

    """
    # TODO: check missing value consistency with GRIB2 (currently comparisons are
    #       done with np.nan)
    #       check that p_field is the pressure field, given in Pa (can only be done
    #       if attributes are consequently set)
    #       check that field and p_field are compatible (have the same
    #       dimensions and sizes)
    #       print warn message if result contains missing values

    # Initializations
    # ... supported interpolation modes
    interpolation_modes = ("linear_in_p", "linear_in_lnp", "nearest_sfc")
    if mode not in interpolation_modes:
        raise RuntimeError("interpolate_k2p: unknown mode", mode)
    # ... supported tc units and corresponding conversion factors to Pa
    p_tc_unit_conversions = dict(Pa=1.0, hPa=100.0)
    if p_tc_units not in p_tc_unit_conversions.keys():
        raise RuntimeError(
            "interpolate_k2p: unsupported value of p_tc_units", p_tc_units
        )
    # ... supported range of pressure tc values (in Pa)
    p_tc_min = 1.0
    p_tc_max = 120000.0

    # Define vertical target coordinates (tc)
    tc_factor = p_tc_unit_conversions[p_tc_units]
    tc_values = np.array(sorted(p_tc_values)) * tc_factor
    if np.any((tc_values < p_tc_min) | (tc_values > p_tc_max)):
        raise RuntimeError(
            "interpolate_k2p: target coordinate value out of range "
            f"(must be in interval [{p_tc_min}, {p_tc_max}]Pa)"
        )
    tc = TargetCoordinates(
        type_of_level="isobaricInPa",
        values=tc_values.tolist(),
        attrs=TargetCoordinatesAttrs(
            units="Pa",
            positive="down",
            standard_name="air_pressure",
            long_name="pressure",
        ),
    )

    # Check that typeOfLevel is supported and equal for both field and p_field
    if field.vcoord_type != "model_level":
        raise RuntimeError(
            "interpolate_k2p: field to interpolate must be defined on model levels"
        )
    if p_field.vcoord_type != "model_level":
        raise RuntimeError(
            "interpolate_k2p: pressure field must be defined on model levels"
        )
    # Check that dimensions are the same for field and p_field
    if field.origin_z != p_field.origin_z:
        raise RuntimeError(
            "interpolate_k2p: field and p_field must have equal vertical staggering"
        )

    # Prepare output field field_on_tc on target coordinates
    field_on_tc = init_field_with_vcoord(field.broadcast_like(p_field), tc, np.nan)

    # Interpolate
    # ... prepare interpolation
    pkm1 = p_field.shift(z=1)
    fkm1 = field.shift(z=1)

    # ... loop through tc values
    for tc_idx, p0 in enumerate(tc_values):
        # ... find the 3d field where pressure is > p0 on level k
        # and was <= p0 on level k-1
        p2 = p_field.where((p_field > p0) & (pkm1 <= p0))
        # ... extract the index k of the vertical layer at which p2 adopts its minimum
        #     (corresponds to search from top of atmosphere to bottom)
        # ... note that if the condition above is not fulfilled, minind will
        # be set to k_top
        minind = p2.fillna(p_tc_max).argmin(dim="z")
        # ... extract pressure and field at level k
        p2 = p2[{"z": minind}]
        f2 = field[{"z": minind}]
        # ... extract pressure and field at level k-1
        # ... note that f1 and p1 are both undefined, if minind equals k_top
        f1 = fkm1[{"z": minind}]
        p1 = pkm1[{"z": minind}]

        # ... compute the interpolation weights
        if mode == "linear_in_p":
            # ... note that p1 is undefined, if minind equals k_top, so ratio will
            # be undefined
            ratio = (p0 - p1) / (p2 - p1)

        if mode == "linear_in_lnp":
            # ... note that p1 is undefined, if minind equals k_top, so ratio will
            #  be undefined
            ratio = (np.log(p0) - np.log(p1)) / (np.log(p2) - np.log(p1))

        if mode == "nearest_sfc":
            # ... note that by construction, p2 is always defined;
            #     this operation sets ratio to 0 if p1 (and by construction also f1)
            #     is undefined; therefore, the interpolation formula below works
            #     correctly also in this case
            ratio = xr.where(np.abs(p0 - p1) >= np.abs(p0 - p2), 1.0, 0.0)

        # ... interpolate and update field_on_tc
        field_on_tc[{"z": tc_idx}] = (1.0 - ratio) * f1 + ratio * f2

    return field_on_tc


def interpolate_k2theta(
    field: xr.DataArray,
    mode: Literal["low_fold", "high_fold", "undef_fold"],
    th_field: xr.DataArray,
    th_tc_values: Sequence[float],
    th_tc_units: Literal["K", "cK"],
    h_field: xr.DataArray,
) -> xr.DataArray:
    """Interpolate a field from model levels to potential temperature coordinates.

       Example for vertical interpolation to isosurfaces of a target field
       that is no monotonic function of height.

    Parameters
    ----------
    field : xarray.DataArray
        field to interpolate (only typeOfLevel="generalVerticalLayer" is supported)
    mode : str
        interpolation algorithm, one of {"low_fold", "high_fold", "undef_fold"}
    th_field : xarray.DataArray
        potential temperature theta on k levels in K
        (only typeOfLevel="generalVerticalLayer" is supported)
    th_tc_values : list of float
        target coordinate values
    th_tc_units : str
        target coordinate units
    h_field : xarray.DataArray
        height on k levels (only typeOfLevel="generalVerticalLayer" is supported)

    Returns
    -------
    field_on_tc : xarray.DataArray
        field on target (i.e., theta) coordinates

    """
    # TODO: check missing value consistency with GRIB2
    #       (currently comparisons are done with np.nan)
    #       check that th_field is the theta field, given in K
    #       (can only be done if attributes are consequently set)
    #       check that field, th_field, and h_field are compatible
    #       print warn message if result contains missing values

    # ATTENTION: the attribute "positive" is not set for generalVerticalLayer
    #            we know that for COSMO it would be defined as positive:"down";
    #            for the time being,
    #            we explicitly use the height field on model mid layer
    #            surfaces as auxiliary field

    # Parameters
    # ... supported folding modes
    folding_modes = ("low_fold", "high_fold", "undef_fold")
    if mode not in folding_modes:
        raise RuntimeError("interpolate_k2theta: unsupported mode", mode)

    # ... supported tc units and corresponding conversion factor to K
    # (i.e. to the same unit as theta); according to GRIB2
    #     isentropic surfaces are coded in K; fieldextra codes
    #     them in cK for NetCDF (to be checked)
    th_tc_unit_conversions = dict(K=1.0, cK=0.01)
    if th_tc_units not in th_tc_unit_conversions.keys():
        raise RuntimeError(
            "interpolate_k2theta: unsupported value of th_tc_units", th_tc_units
        )
    # ... supported range of tc values (in K)
    th_tc_min = 1.0
    th_tc_max = 1000.0
    # ... tc values outside range of meaningful values of height,
    # used in tc interval search (in m amsl)
    h_min = -1000.0
    h_max = 100000.0

    # Define vertical target coordinates
    # Sorting cannot be exploited for optimizations, since theta is
    # not monotonous wrt to height tc values are stored in K
    tc_values = np.array(th_tc_values) * th_tc_unit_conversions[th_tc_units]
    if np.any((tc_values < th_tc_min) | (tc_values > th_tc_max)):
        raise RuntimeError(
            "interpolate_k2theta: target coordinate value "
            f"out of range (must be in interval [{th_tc_min}, {th_tc_max}]K)"
        )
    tc = TargetCoordinates(
        type_of_level="theta",
        values=tc_values.tolist(),
        attrs=TargetCoordinatesAttrs(
            units="K",
            positive="up",
            standard_name="air_potential_temperature",
            long_name="potential temperature",
        ),
    )

    # Check that typeOfLevel is supported and equal for field, th_field, and h_field
    if field.vcoord_type != "model_level" or field.origin_z != 0.0:
        raise RuntimeError(
            "interpolate_k2theta: field to interpolate must "
            "be defined on model_level layers"
        )
    if th_field.vcoord_type != "model_level" or th_field.origin_z != 0.0:
        raise RuntimeError(
            "interpolate_k2theta: theta field must be defined on model_level layers"
        )
    if h_field.vcoord_type != "model_level" or h_field.origin_z != 0.0:
        raise RuntimeError(
            "interpolate_k2theta: height field must be defined on model_level layers"
        )

    # Prepare output field field_on_tc on target coordinates
    field_on_tc = init_field_with_vcoord(field.broadcast_like(th_field), tc, np.nan)

    # Interpolate
    # ... prepare interpolation
    thkm1 = th_field.shift(z=1)
    fkm1 = field.shift(z=1)

    # ... loop through tc values
    for tc_idx, th0 in enumerate(tc.values):
        folding_coord_exception = xr.full_like(h_field[{"z": 0}], False)
        # ... find the height field where theta is >= th0 on level k and was <= th0
        #     on level k-1 or where theta is <= th0 on level k
        #     and was >= th0 on level k-1
        h = h_field.where(
            ((th_field >= th0) & (thkm1 <= th0)) | ((th_field <= th0) & (thkm1 >= th0))
        )
        if mode == "undef_fold":
            # ... find condition where more than one interval is found, which
            # contains the target coordinate value
            folding_coord_exception = xr.where(h.notnull(), 1.0, 0.0).sum(dim=["z"])
            folding_coord_exception = folding_coord_exception.where(
                folding_coord_exception > 1.0
            ).notnull()
        if mode in ("low_fold", "undef_fold"):
            # ... extract the index k of the smallest height at which
            # the condition is fulfilled
            tcind = h.fillna(h_max).argmin(dim="z")
        if mode == "high_fold":
            # ... extract the index k of the largest height at which the condition
            # is fulfilled
            tcind = h.fillna(h_min).argmax(dim="z")

        # ... extract theta and field at level k
        th2 = th_field[{"z": tcind}]
        f2 = field[{"z": tcind}]
        # ... extract theta and field at level k-1
        f1 = fkm1[{"z": tcind}]
        th1 = thkm1[{"z": tcind}]

        # ... compute the interpolation weights
        ratio = xr.where(np.abs(th2 - th1) > 0, (th0 - th1) / (th2 - th1), 0.0)

        # ... interpolate and update field_on_tc
        field_on_tc[{"z": tc_idx}] = xr.where(
            folding_coord_exception, np.nan, (1.0 - ratio) * f1 + ratio * f2
        )

    return field_on_tc


def interpolate_k2any(
    field: xr.DataArray,
    mode: Literal["low_fold", "high_fold"],
    tc_field: xr.DataArray,
    tc: TargetCoordinates,
    h_field: xr.DataArray,
) -> xr.DataArray:
    """Interpolate a field from model levels to coordinates w.r.t. an arbitrary field.

    Example for vertical interpolation to isosurfaces of a target field
    that is no monotonic function of height.

    Parameters
    ----------
    field : xarray.DataArray
        field to interpolate (only typeOfLevel="generalVerticalLayer" is supported)
    mode : str
        interpolation algorithm, one of {"low_fold", "high_fold"}
    tc_field : xarray.DataArray
        target field
        (only typeOfLevel="generalVerticalLayer" is supported)
    tc : TargetCoordinates
        target coordinate definition
    h_field : xarray.DataArray
        height on k levels (only typeOfLevel="generalVerticalLayer" is supported)

    Returns
    -------
    field_on_tc : xarray.DataArray
        field on target coordinates

    """
    modes = ("low_fold", "high_fold")
    if mode not in modes:
        raise ValueError(f"Unsupported mode: {mode}")

    for f in (field, tc_field, h_field):
        if f.vcoord_type != "model_level" or f.origin_z != 0.0:
            raise ValueError("Input fields must be defined on full model levels")

    # ... tc values outside range of meaningful values of height,
    # used in tc interval search (in m amsl)
    h_min = -1000.0
    h_max = 100000.0

    # Prepare output field field_on_tc on target coordinates
    field_on_tc = init_field_with_vcoord(field.broadcast_like(tc_field), tc, np.nan)

    # Interpolate
    # ... prepare interpolation
    tckm1 = tc_field.shift(z=1)
    fkm1 = field.shift(z=1)

    for tc_idx, value in enumerate(tc.values):
        # ... find the height field where target is >= value on level k and was <= value
        #     on level k-1 or where target is <= value on level k
        #     and was >= value on level k-1
        h = h_field.where(
            ((tc_field >= value) & (tckm1 <= value))
            | ((tc_field <= value) & (tckm1 >= value))
        )
        if mode == "low_fold":
            # ... extract the index k of the smallest height at which
            # the condition is fulfilled
            tcind = h.fillna(h_max).argmin(dim="z")
        if mode == "high_fold":
            # ... extract the index k of the largest height at which the condition
            # is fulfilled
            tcind = h.fillna(h_min).argmax(dim="z")

        # ... extract target and field at level k
        t2 = tc_field[{"z": tcind}]
        f2 = field[{"z": tcind}]
        # ... extract target and field at level k-1
        f1 = fkm1[{"z": tcind}]
        t1 = tckm1[{"z": tcind}]

        # ... compute the interpolation weights
        ratio = xr.where(np.abs(t2 - t1) > 0, (value - t1) / (t2 - t1), 0.0)

        # ... interpolate and update field_on_tc
        field_on_tc[{"z": tc_idx}] = (1.0 - ratio) * f1 + ratio * f2

    return field_on_tc
