import jax
import jax.numpy as jnp
import jax.scipy as jsp

# --------------------------------
# gaussian function
def gaussian(x, mu, sigma):
    return jnp.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * jnp.sqrt(2 * jnp.pi))

# --------------------------------
# gaussian function normalized to 1 at the peak
def gaussian_normax(x, mu, sigma):
    return jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)

# --------------------------------
# construct rotation matrices given yaw, pitch, roll angles
def get_Ms(theta_0, theta_1, theta_2):

    c0 = jnp.cos(theta_0)
    s0 = jnp.sin(theta_0)
    c1 = jnp.cos(theta_1)
    s1 = jnp.sin(theta_1)
    c2 = jnp.cos(theta_2)
    s2 = jnp.sin(theta_2)

    M1 = jnp.array([[c0, -s0, 0],
                [s0,  c0, 0],
                [  0,   0, 1]])
    M2 = jnp.array([[ c1, 0, s1],
                [  0, 1,  0],
                [-s1, 0, c1]])
    M3 = jnp.array([[1,   0,    0],
                [0,  c2, -s2],
                [0,  s2,  c2]])
    return M1, M2, M3

# --------------------------------
# rotate coordinates given rotation matrices
def rotate(Xo, Yo, Zo, M1, M2, M3):

    X1 = M1[0,0]*Xo + M1[0,1]*Yo + M1[0,2]*Zo
    Y1 = M1[1,0]*Xo + M1[1,1]*Yo + M1[1,2]*Zo
    Z1 = M1[2,0]*Xo + M1[2,1]*Yo + M1[2,2]*Zo

    X2 = M2[0,0]*X1 + M2[0,1]*Y1 + M2[0,2]*Z1
    Y2 = M2[1,0]*X1 + M2[1,1]*Y1 + M2[1,2]*Z1
    Z2 = M2[2,0]*X1 + M2[2,1]*Y1 + M2[2,2]*Z1

    X = M3[0,0]*X2 + M3[0,1]*Y2 + M3[0,2]*Z2
    Y = M3[1,0]*X2 + M3[1,1]*Y2 + M3[1,2]*Z2
    Z = M3[2,0]*X2 + M3[2,1]*Y2 + M3[2,2]*Z2

    return X, Y, Z

# --------------------------------
# derotate coordinates given rotation matrices
def derotate(X, Y, Z, M1, M2, M3):

    Minv3 = jnp.linalg.inv(M3)
    Minv2 = jnp.linalg.inv(M2)
    Minv1 = jnp.linalg.inv(M1)

    X2 = Minv3[0,0]*X + Minv3[0,1]*Y + Minv3[0,2]*Z
    Y2 = Minv3[1,0]*X + Minv3[1,1]*Y + Minv3[1,2]*Z
    Z2 = Minv3[2,0]*X + Minv3[2,1]*Y + Minv3[2,2]*Z

    X1 = Minv2[0,0]*X2 + Minv2[0,1]*Y2 + Minv2[0,2]*Z2
    Y1 = Minv2[1,0]*X2 + Minv2[1,1]*Y2 + Minv2[1,2]*Z2
    Z1 = Minv2[2,0]*X2 + Minv2[2,1]*Y2 + Minv2[2,2]*Z2

    Xo = Minv1[0,0]*X1 + Minv1[0,1]*Y1 + Minv1[0,2]*Z1
    Yo = Minv1[1,0]*X1 + Minv1[1,1]*Y1 + Minv1[1,2]*Z1
    Zo = Minv1[2,0]*X1 + Minv1[2,1]*Y1 + Minv1[2,2]*Z1

    return Xo, Yo, Zo

# --------------------------------
# smooth an image with a Gaussian kernel
def smooth1(Z, sigma=4):
    radius = round(4. * sigma)

    x = jnp.linspace(-radius, radius, 2*radius + 1)
    window = jsp.stats.norm.pdf(x, loc=0, scale=sigma) * jsp.stats.norm.pdf(x[:, None], loc=0, scale=sigma)
    window /= jnp.sum(window)

    smooth_image = jax.vmap(jsp.signal.convolve, in_axes=(2, None, None), out_axes=2)(Z, window, 'same')

    return smooth_image

# -------------------------------
def ppv_to_fits(ppv, vchans, angular_size, ra=760.7, dec=251.8, projection='GLS', fname='ppv.fits'):
    from astropy.io import fits

    hdu = fits.PrimaryHDU(ppv.transpose(2, 1, 0))
    hdul = fits.HDUList([hdu])

    hdr = hdul[0].header
    hdr['DATAMIN'] = ppv.min()
    hdr['DATAMAX'] = ppv.max()
    hdr['CTYPE1'] = f"RA--{projection}"
    hdr['CRVAL1'] = 0.0
    hdr['CRPIX1'] = ppv.shape[0] // 2 + 1
    hdr['CDELT1'] = 1.0  # arbitrary units

    hdr['CTYPE2'] = f"DEC--{projection}"

    hdr['CTYPE3'] = "VRAD"

    hdul.writeto(fname, overwrite=True)
