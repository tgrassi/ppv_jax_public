import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------
# get line velocities and strengths for a given molecule
def get_lines(molecule):

    if molecule == "NH2D":
        from data.struct_NH2D import freq_dict_cen, freq_dict, line_strength_dict

        clight = 2.99792458e5  # km/s

        freq0 = freq_dict_cen["p-1_11-1_01"]

        vels = []
        lss = []
        for k, ff in freq_dict.items():
            if k.startswith("p-1_11-1_01"):
                ls = line_strength_dict[k]
                vel = (freq0 - ff) / freq0 * clight
                vels.append(vel)
                lss.append(ls)

    elif molecule == "N2D+":
        from data.struct_N2Dp import freq_dict_cen, voff_lines_dict, line_strength_dict

        freq0 = freq_dict_cen["J2-1"]

        vels = []
        lss = []
        for k, dv in voff_lines_dict.items():
            if k.startswith("J2-1"):
                ls = line_strength_dict[k]
                vel = dv
                vels.append(vel)
                lss.append(ls)
    else:
        raise ValueError("Molecule not recognized " + molecule)

    vels = np.array(vels)
    lss = np.array(lss)

    lss /= np.max(lss)

    return vels, lss, freq0

# --------------------------------
# plot spectrum given velocity profile and intensity profile
def plot_spectrum(vprof, Iprof, sigma, vchans, filename=None, vrange=None):

    spectrum = jnp.zeros_like(vchans)

    for v, I in zip(vprof, Iprof):
        spectrum += I * jnp.exp(-0.5 * ((vchans - v) / sigma) ** 2) / (sigma * jnp.sqrt(2 * jnp.pi))

    plt.figure(figsize=(6,4))
    plt.plot(vchans, spectrum)
    plt.axvline(0., color='k', ls='--')
    if vrange is not None:
        plt.xlim(vrange[0], vrange[1])
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


