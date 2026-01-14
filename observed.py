from astropy.io import fits
import numpy as np
from scipy.interpolate import interp1d

def load_fits_data(fits_file, freq0, voff):
    vw_obs = fits.getdata(fits_file)#.astype(jnp.float32)
    vw_obs = vw_obs.transpose(2, 1, 0)

    nx, ny, nv = vw_obs.shape

    hdr = fits.getheader(fits_file)
    vhdr = (hdr['CRVAL3'] + (np.arange(nv) - hdr['CRPIX3']) * hdr['CDELT3']) #.astype(jnp.float32)
    vhdr = vhdr / 1e3  # to km/s

    vhdr_org = vhdr.copy()

    restfreq = hdr['RESTFREQ']

    clight = 2.99792458e5  # km/s

    ff = restfreq * (vhdr / clight + 1e0)

    ff = ff - restfreq + freq0

    vhdr = (ff - restfreq) / restfreq * clight - voff

    if vhdr[1] < vhdr[0]:
        vhdr = vhdr[::-1]
        vw_obs = vw_obs[:, :, ::-1]

    return vhdr, vw_obs, restfreq, vhdr_org

def get_observed_ppv(fits_file, freq0, vchans, vbulk=0e0):
    vhdr, vw_obs, _, _ = load_fits_data(fits_file, freq0, vbulk)

    finterp = interp1d(vhdr, vw_obs, kind='linear', axis=-1, bounds_error=False, fill_value=0e0)

    ppv_observed = finterp(vchans)

    ppv_observed /= np.max(ppv_observed)

    return ppv_observed, vhdr