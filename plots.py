import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np


# ---------------------------------
def plot_profiles(models, molecules, fname=None):
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))

    if not isinstance(models, list):
        models = [models]

    lss = ['-', '--', ':', '-.']

    vmin = nmin = np.inf
    vmax = nmax = -np.inf

    for imod, mod in enumerate(models):

        # reset color cycle
        for ax in axs.flatten():
            ax.set_prop_cycle(None)

        nmod = len(mod)

        for i in range(nmod):
            Xo, Yo, Zo, vx, vy, vz, ngas = mod[i, :, :, ::, :]

            vmin = min(vmin, jnp.min(vy))
            vmax = max(vmax, jnp.max(vy))

            nx, ny, nz = Xo.shape
            ax = axs[0, 0]
            ax.plot(Xo[:, ny//2, nz//2], vy[:, ny//2, nz//2], label=molecules[i], ls=lss[imod])

            ax = axs[0, 1]
            ax.plot(Yo[nx//2, :, nz//2], vy[nx//2, :, nz//2], ls=lss[imod])

            ax = axs[0, 2]
            ax.plot(Zo[nx//2, ny//2, :], vy[nx//2, ny//2, :], ls=lss[imod])

        for i in range(nmod):
            Xo, Yo, Zo, vx, vy, vz, ngas = mod[i, :, :, ::, :]

            nmin = min(nmin, jnp.min(ngas))
            nmax = max(nmax, jnp.max(ngas))

            nx, ny, nz = Xo.shape
            ax = axs[1, 0]
            ax.plot(Xo[:, ny//2, nz//2], ngas[:, ny//2, nz//2], ls=lss[imod])

            ax = axs[1, 1]
            ax.plot(Yo[nx//2, :, nz//2], ngas[nx//2, :, nz//2], ls=lss[imod])

            ax = axs[1, 2]
            ax.plot(Zo[nx//2, ny//2, :], ngas[nx//2, ny//2, :], ls=lss[imod])

    for i in range(3):
        axs[0, i].set_ylim(vmin, vmax)
        axs[1, i].set_ylim(nmin*0.9, nmax*1.1)
        if i == 0:
            axs[0, i].set_ylabel("Velocity (km/s)")
            axs[1, i].set_ylabel("Gas Density (arbitrary units)")
        else:
            axs[0, i].set_yticklabels([])
            axs[1, i].set_yticklabels([])

        axs[0, i].set_xticks([])
        for j in range(2):
            axs[j, i].axvline(0.0, color='k', ls=':', lw=1, alpha=0.5)

    axs[0, 0].legend(ncols=len(models), loc='upper right', fontsize=10)
    axs[1, 0].set_xlabel("x (y=0, z=0)")
    axs[1, 1].set_xlabel("y (x=0, z=0)")
    axs[1, 2].set_xlabel("z (x=0, y=0)")

    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
    plt.show()


# ---------------------------------
def plot_channels(ppvs, vchans, molecules, nchans=30, fname=None):

    fig, axs = plt.subplots(2, nchans, figsize=(nchans, 2))
    vmin = jnp.min(ppvs)
    vmax = jnp.max(ppvs)

    emax = jnp.max(ppvs, axis=(0, 1, 2))
    imax = jnp.argmax(emax)
    imin = imax - nchans//2

    print("Velocity channel max", vchans[imax], "km/s")

    for i in range(2):
        axs[i, 0].text(-0.5, 0.5, molecules[i], fontsize=12, ha='right', va='center', transform=axs[i, 0].transAxes)
        for j in range(nchans):
            ax = axs[i, j]
            ax.pcolor(ppvs[i, ..., imin + j], vmin=vmin, vmax=vmax, cmap='inferno')
            ax.set_aspect('equal')
            ax.axis('off')

    nv = vchans.size
    print(f"Plotting {nchans} channels out of {nv} in [{vchans[imin]:.2f}-{vchans[imax]:.2f}] km/s")

    if fname is not None:
        plt.savefig(fname)
    plt.show()

# ---------------------------------
def plot_spectra(ppvs, vchans, molecules, ppvs2=None, fname=None, labels=None):
    nmodels, nx, nz, _ = ppvs.shape

    plt.figure(figsize=(10, 4))
    for i in range(nmodels):
        postfix = ""
        if labels is not None:
            postfix = f" ({labels[0]})"
        p = plt.plot(vchans, ppvs[i, nx//2, nz//2, :], label=f"{molecules[i]}{postfix}")
        if ppvs2 is not None:
            postfix2 = ""
            if labels is not None:
                postfix2 = f" ({labels[1]})"
            plt.plot(vchans, ppvs2[i, nx//2, nz//2, :], lw=1, marker='.', alpha=0.5, color=p[0].get_color(), label=f"{molecules[i]}{postfix2}")

    plt.xlabel("Velocity (km/s)")
    plt.ylabel("Intensity")

    ncols = 1
    if ppvs2 is not None:
        ncols = 2
    plt.legend(ncols=ncols)
    plt.title("Spectra at x,z = center")
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
    plt.show()

# ---------------------------------
def plot_slice(models, molecules, fname=None, titles=None, slice='xy'):

    if not isinstance(models, list):
        models = [models]

    nmod = len(models)
    fig, axs = plt.subplots(1, nmod, figsize=(5*nmod, 8))

    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    nmax = -np.inf
    for model in models:
        for i in range(model.shape[0]):
            _, _, _, _, _, _, ngas = model[i, :, :, ::, :]
            nmax = max(nmax, jnp.max(ngas))

    for i, model in enumerate(models):
        step = model.shape[3]//16

        Xo, Yo, Zo, vx, vy, vz, ngas = model[0, :, :, ::step, :]
        Xo1, Yo1, Zo1, vx1, vy1, vz1, ngas1 = model[1, :, :, ::step, :]
        vbulk_pseudo = np.mean(vy)
        vy -= vbulk_pseudo
        vy1 -= vbulk_pseudo

        xcut = ngas.shape[0]//2
        zcut = ngas.shape[2]//2

        ax = axs[i]

        if slice == 'xy':
            Ca0 = Xo[:, :, zcut]
            Cb0 = Yo[:, :, zcut]
            Ca1 = Xo1[:, :, zcut]
            Cb1 = Yo1[:, :, zcut]
            ngas0 = ngas[:, :, zcut]
            va0 = vx[:, :, zcut]
            vb0 = vy[:, :, zcut]
            va1 = vx1[:, :, zcut]
            vb1 = vy1[:, :, zcut]
            labela = 'X'
            labelb = 'Y (LOS)'
        elif slice == 'zy':
            Ca0 = Zo[xcut, :, :]
            Cb0 = Yo[xcut, :, :]
            Ca1 = Zo1[xcut, :, :]
            Cb1 = Yo1[xcut, :, :]
            ngas0 = ngas[xcut, :, :]
            va0 = vz[xcut, :, :]
            vb0 = vy[xcut, :, :]
            va1 = vz1[xcut, :, :]
            vb1 = vy1[xcut, :, :]
            labela = 'Z'
            labelb = 'Y (LOS)'
        else:
            raise ValueError("Slice not recognized: " + slice)

        ax.pcolor(Ca0, Cb0, ngas0, cmap='Blues', vmin=0, vmax=nmax)
        #plt.colorbar(label=f"n({molecules[0]})")
        ax.quiver(Ca0, Cb0, va0, vb0, color='tab:cyan', label=molecules[0])
        ax.quiver(Ca1, Cb1, va1, vb1, color='tab:orange', label=molecules[1])
        ax.axvline(0, color='k', linestyle='--', label='LOS', alpha=0.5)
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        ax.set_xlabel(labela)
        ax.set_xlim(Ca0.min(), Ca0.max())
        ax.set_ylim(Cb0.min(), Cb0.max())
        if titles is not None:
           ax.set_title(titles[i])

    axs[0].set_ylabel(labelb)
    for i in range(1, nmod):
        axs[i].set_yticks([])

    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()

# ---------------------------------
def plot_emission(models, emissions, vchans, molecules, fname=None, xpos=0., zpos=0., vrange=None):

    nmols = len(molecules)
    figs, axs = plt.subplots(1, nmols, figsize=(5*nmols, 5))

    em_max = jnp.max(emissions)
    ngas_max = np.max([models[i][6].max() for i in range(nmols)])
    for i in range(nmols):
        ax = axs[i]
        em = emissions[i]

        Xo, Yo, Zo, vx, vy, vz, ngas = models[i]

        ix = np.argmin(np.abs(Xo[:, 0, 0] - xpos))
        iz = np.argmin(np.abs(Zo[0, 0, :] - zpos))

        ax.contourf(vchans, Yo[ix, :, iz], em[ix, :, iz, :] / em_max, vmin=0, vmax=1, cmap='Greys_r')

        ax.plot(vy[ix, :, iz], Yo[ix, :, iz], label='$v_y(y)$')
        ax.scatter(vy[ix, :, iz], Yo[ix, :, iz], color='tab:orange', s=(ngas[ix, :, iz]/ngas_max)**2*50, alpha=0.5, label='$n(y)$')
        if i > 0:
            ax.set_yticks([])
        else:
            ax.set_ylabel("y (LOS)")
        ax.set_xlabel("Velocity (km/s)")
        ax.set_title(f"Emission map for {molecules[i]}")
        if vrange is not None:
            ax.set_xlim(vrange)
        ax.legend()

    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()

# ---------------------------------
def plot_comparison(ppvs_target, ppvs_optimized, vchans, molecules, nchans=30, fname=None):

    fig, axs = plt.subplots(6, nchans, figsize=(nchans, 6))
    vmin = jnp.min(ppvs_target)
    vmax = jnp.max(ppvs_target)

    emax = jnp.max(ppvs_target, axis=(0, 1, 2))
    imax = jnp.argmax(emax)

    print("Velocity channel max", vchans[imax])

    diff = (ppvs_optimized - ppvs_target) / jnp.max(ppvs_target)
    val = jnp.max(jnp.abs(diff))
    print("Max difference:", val)

    for i in range(6):
        axs[i, 0].text(-0.5, 0.5, molecules[i//3], fontsize=12, ha='right', va='center', transform=axs[i, 0].transAxes)
        for j in range(nchans):
            ax = axs[i, j]
            if i % 3 == 0:
                ax.pcolor(ppvs_target[i//3, ..., imax - nchans//2 + j], vmin=vmin, vmax=vmax, cmap='inferno')
            elif i % 3 == 1:
                ax.pcolor(ppvs_optimized[i//3, ..., imax - nchans//2 + j], vmin=vmin, vmax=vmax, cmap='inferno')
            else:
                ax.pcolor(diff[i//3, ..., imax - nchans//2 + j], vmin=-val, vmax=val, cmap='seismic')
            ax.set_aspect('equal')
            ax.axis('off')

    nv = vchans.size
    plt.suptitle(f"Target, Optimized, and Difference PPV Cubes ({nchans} channels out of {nv})")
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
    plt.show()
