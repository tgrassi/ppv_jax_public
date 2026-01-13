import jax.numpy as jnp
import jax
from utils import gaussian, gaussian_normax, rotate, get_Ms, smooth1, derotate

"""
Compute position-velocity cube for a gas kinematic model.

This function models gas kinematics in a rotating system, including density
distribution, velocity field, and radiative transfer effects. It computes
the intensity distribution in position-position-velocity space by integrating
along the line of sight with absorption.

Args:
    pp (dict): Parameter dictionary containing:
        - mff: Infall matrix component
        - mrot: Rotation matrix component
        - xoffset: Offset in x direction (indexed by imodel)
        - zoffset: Offset in z direction (indexed by imodel)
        - vbulk: Bulk velocity in y direction (LOS)
        - v0: Velocity scale Gaussian radial profile (indexed by imodel)
        - r0_v: Center of Gaussian velocity profile (indexed by imodel)
        - sigma_v: Dispersion of Gaussian velocity profile (indexed by imodel)
        - ngas0: Gas density Gaussian radial profile (indexed by imodel)
        - r0_ngas: Center of Gaussian gas density profile (indexed by imodel)
        - sigma_ngas: Width of Gaussian gas density profile (indexed by imodel)
        - sigma_turb: Microturbulence and thermal velocity dispersion (indexed by imodel)
        - sigmadx: Opacity per unit length (indexed by imodel)
        - theta0, theta1, theta2: Rotation angles (yaw, pitch, roll)
        - zratio: Ellipticity parameter
        - asym: Asymmetry amplitude (indexed by imodel)
        - asym_theta: Asymmetry phase (indexed by imodel)

    pa (dict): Model arguments dictionary containing:
        - Iprof: Intensity profile for velocity channels (indexed by imodel)
        - vprof: Velocity profile for channels (indexed by imodel)
        - vchans: Velocity channel centers

    sigma_beam (float, optional): Beam smoothing parameter. Default: 1
    imodel (int, optional): Model index. Default: 0

Returns:
    tuple: A tuple containing:
        - ppv (ndarray): Position-position-velocity cube (x, z, v)
        - em (ndarray): Emission cube before line-of-sight integration (x, y, z, v)
        - v (ndarray): Velocity magnitude model cube (x, y, z)
        - stack (ndarray): Model grid with coordinates and velocity components (x, y, z, vx, vy, vz, ngas)
"""

def f(pp, pa, sigma_beam=1, imodel=0):

    # unpack parameters for readability
    mff = pp["mff"]
    mrot = pp["mrot"]
    xoffset = pp["xoffset"][imodel]
    zoffset = pp["zoffset"][imodel]
    vbulk = pp["vbulk"]
    v0 = pp["v0"][imodel]
    r0_v = pp["r0_v"][imodel]
    sigma_v = pp["sigma_v"][imodel]
    ngas0 = pp["ngas0"][imodel]
    r0_ngas = pp["r0_ngas"][imodel]
    sigma_ngas = pp["sigma_ngas"][imodel]
    sigma_turb = pp["sigma_turb"][imodel]
    sigmadx = pp["sigmadx"][imodel]
    theta0 = pp["theta0"]
    theta1 = pp["theta1"]
    theta2 = pp["theta2"]
    zratio = pp["zratio"]
    asym = pp["asym"][imodel]
    asym_theta = pp["asym_theta"][imodel]

    sigmadx = 1e1**sigmadx

    # unpack model arguments
    Iprof = pa["Iprof"][imodel]
    vprof = pa["vprof"][imodel]
    vchans = pa["vchans"]

    # construct the versor field matrix
    m = jnp.array([mff, -mrot, 0e0,
                   mrot, mff, 0e0,
                   0e0, 0e0, mff])

    # create grid
    x = jnp.linspace(-1., 1., 16*sigma_beam)
    y = jnp.linspace(-1., 1., 32) # <<<<< points along LOS
    z = jnp.linspace(-1., 1., 16*sigma_beam)

    # meshgrid in object frame
    Xo, Yo, Zo = jnp.meshgrid(x, y, z, indexing='ij')

    # rotate grid to model frame (including offsets in the plane of the sky)
    M1, M2, M3 = get_Ms(theta0, theta1, theta2)
    Xrot, Yrot, Zrot = rotate(Xo - xoffset, Yo, Zo - zoffset, M1, M2, M3)

    # compute radius
    r = jnp.sqrt(Xrot**2 + Yrot**2 + Zrot**2 + 1e-10)

    # compute elliptical radius
    r_el = jnp.sqrt(Xrot**2 + Yrot**2 + zratio**2 * Zrot**2 + 1e-10)

    # compute gas density
    ngas = ngas0 * gaussian_normax(r_el, r0_ngas, sigma_ngas)

    # compute versor field
    ux = m[0] * Xrot + m[1] * Yrot + m[2] * Zrot
    uy = m[3] * Xrot + m[4] * Yrot + m[5] * Zrot
    uz = m[6] * Xrot + m[7] * Yrot + m[8] * Zrot

    # derotate versor field (because we rotated the grid)
    ux, uy, uz = derotate(ux, uy, uz, M1, M2, M3)

    # normalize versor field
    uu = jnp.sqrt(ux**2 + uy**2 + uz**2 + 1e-10)
    uux = ux / uu
    uuy = uy / uu
    uuz = uz / uu

    # compute gas velocity applying gaussian radial profile
    vgas = v0 * gaussian_normax(r, r0_v, sigma_v)

    # apply asymmetry
    vgas *= 1e0 + asym * jnp.sin(jnp.arctan2(Yrot, Xrot) + asym_theta)

    # compute velocity components
    vx = -uux * vgas
    vy = -uuy * vgas
    vz = -uuz * vgas

    # compute velocity
    v = jnp.sqrt(vx**2 + vy**2 + vz**2 + 1e-10)

    # add bulk velocity
    vy += vbulk

    # add profile velocity channels
    sm = vy[..., None] + vprof[None, None, None, :]

    # assume lines are broadened by turbulence and thermal motions using a gaussian kernel
    gg = gaussian(vchans[None, None, None, None, :], sm[..., None], sigma_turb) #[..., None, None])
    Ichans_gp = Iprof[None, None, None, :, None] * gg
    Ichans_g = Ichans_gp.sum(axis=3)

    # intensity is proportional to density and intensity of spectral channels
    nlos_ww = Ichans_g * ngas[..., None]

    # compute tau
    tau = jnp.cumsum(nlos_ww, axis=1) * sigmadx

    # compute absorption
    abs = jnp.exp(-tau)

    # compute emission
    em = nlos_ww * abs

    # integrate along los
    ppv = jnp.trapezoid(em, y, axis=1)

    # smooth to mimic beam
    if sigma_beam > 1:
        ppv = smooth1(ppv, sigma=sigma_beam)[sigma_beam//2::sigma_beam, sigma_beam//2::sigma_beam, :]

    return ppv, em, jnp.stack([Xo, Yo, Zo, vx, vy, vz, ngas])

'''
Get position-position-velocity cubes for multiple models.

This function computes the position-position-velocity (PPV) cubes, emission cubes,
velocity cubes, and model grids for a specified number of models using the provided
parameters and model arguments.

Args:
    params (dict): Parameter dictionary containing model parameters.
    model_args (dict): Model arguments dictionary containing intensity and velocity profiles.
    nmodels (int, optional): Number of models to compute. Default: 2.
    just_ppv (bool, optional): If True, only return the PPV cubes. Default: False.

Returns:
    - ppvs (ndarray): Stack of position-position-velocity cubes for each model.
    - dvs (ndarray): Difference in velocity cubes between models.
    - ems (ndarray): Stack of emission cubes for each model.
    - models (ndarray): Stack of model grids with coordinates and velocity components.

See f documentation for details on the outputs.
'''
def get_ppvs(params, model_args=None, nmodels=2, just_ppv=False):

    ppvs = []
    ems = []
    models = []
    for i in range(nmodels):
        ppv, em, model = f(params, model_args, sigma_beam=1, imodel=i)
        ppvs.append(ppv)
        ems.append(em)
        models.append(model)

    ppvs = jnp.stack(ppvs, axis=0)
    ems = jnp.stack(ems, axis=0)
    models = jnp.stack(models, axis=0)

    if just_ppv:
        return ppvs
    else:
        return ppvs, ems, models

'''
Compute the derivative of the position-position-velocity cubes with respect to a model parameter.

Args:
    params (dict): Parameter dictionary containing model parameters.
    model_args (dict): Model arguments dictionary containing intensity and velocity profiles.
    dparam (str): Name of the parameter to differentiate with respect to.
    nmodels (int, optional): Number of models to compute. Default: 2.

Returns:
        - dy_dxis (ndarray): Derivative of the PPV cubes with respect to the specified parameter (nmodels, nx, nz, nv).

See get_ppvs documentation for details on the outputs.
'''
def df_dparam(params, model_args, dparam, nmodels=2):
    from functools import partial

    get_ppvs_partial = partial(get_ppvs, model_args=model_args, nmodels=nmodels, just_ppv=True)

    tangent = {k: jnp.zeros_like(v) for k, v in params.items()}

    psize = tangent[dparam].size

    dy_dxis = []
    for i in range(nmodels):
        tangent[dparam] = jnp.zeros_like(params[dparam])
        if psize > 1:
            tangent[dparam] = tangent[dparam].at[i].set(1.0)
        else:
            tangent[dparam] = jnp.array(1.0)

        _, dy_dxi = jax.jvp(get_ppvs_partial, (params,), (tangent,))
        dy_dxis.append(dy_dxi[i])

    dy_dxis = jnp.stack(dy_dxis, axis=0)

    return dy_dxis

def get_param(params, param_name, nmodel=0):
    if params[param_name].ndim == 1:
        return params[param_name][nmodel]
    else:
        return params[param_name]
