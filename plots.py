import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np


# ---------------------------------
def plot_profiles(models, molecules):
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    nmod = len(models)

    vmin = np.inf
    vmax = -np.inf
    for i in range(nmod):
        Xo, Yo, Zo, vx, vy, vz, ngas = models[i, :, :, ::, :]

        vmin = min(vmin, jnp.min(vy))
        vmax = max(vmax, jnp.max(vy))

        nx, ny, nz = Xo.shape
        ax = axs[0, 0]
        ax.plot(Xo[:, ny//2, nz//2], vy[:, ny//2, nz//2], label=molecules[i])

        ax = axs[0, 1]
        ax.plot(Yo[nx//2, :, nz//2], vy[nx//2, :, nz//2])

        ax = axs[0, 2]
        ax.plot(Zo[nx//2, ny//2, :], vy[nx//2, ny//2, :])

    nmin = np.inf
    nmax = -np.inf
    for i in range(nmod):
        Xo, Yo, Zo, vx, vy, vz, ngas = models[i, :, :, ::, :]

        nmin = min(nmin, jnp.min(ngas))
        nmax = max(nmax, jnp.max(ngas))

        nx, ny, nz = Xo.shape
        ax = axs[1, 0]
        ax.plot(Xo[:, ny//2, nz//2], ngas[:, ny//2, nz//2])

        ax = axs[1, 1]
        ax.plot(Yo[nx//2, :, nz//2], ngas[nx//2, :, nz//2])

        ax = axs[1, 2]
        ax.plot(Zo[nx//2, ny//2, :], ngas[nx//2, ny//2, :])

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

    axs[0, 0].legend()
    axs[1, 0].set_xlabel("x (y=0, z=0)")
    axs[1, 1].set_xlabel("y (x=0, z=0)")
    axs[1, 2].set_xlabel("z (x=0, y=0)")

    plt.tight_layout()
    plt.show()


# ---------------------------------
def plot_channels(ppvs, vchans, molecules, nchans=30):

    fig, axs = plt.subplots(2, nchans, figsize=(nchans, 2))
    vmin = jnp.min(ppvs)
    vmax = jnp.max(ppvs)

    emax = jnp.max(ppvs, axis=(0, 1, 2))
    imax = jnp.argmax(emax)

    print("Velocity channel max", vchans[imax], "km/s")

    for i in range(2):
        axs[i, 0].text(-0.5, 0.5, molecules[i], fontsize=12, ha='right', va='center', transform=axs[i, 0].transAxes)
        for j in range(nchans):
            ax = axs[i, j]
            ax.pcolor(ppvs[i, ..., imax - nchans//2 + j], vmin=vmin, vmax=vmax, cmap='inferno')
            ax.set_aspect('equal')
            ax.axis('off')

    nv = vchans.size

    plt.suptitle(f"Target PPV Cubes ({nchans} channels out of {nv})", fontsize=16/30.*nchans)
    #plt.tight_layout()
    plt.show()

# ---------------------------------
def plot_spectra(ppvs, vchans, molecules, ppvs2=None):
    nmodels, nx, nz, _ = ppvs.shape

    plt.figure(figsize=(10, 4))
    for i in range(nmodels):
        p = plt.plot(vchans, ppvs[i, nx//2, nz//2, :], label=molecules[i])
        if ppvs2 is not None:
            plt.plot(vchans, ppvs2[i, nx//2, nz//2, :], lw=1, marker='.', alpha=0.5, color=p[0].get_color())

    plt.xlabel("Velocity (km/s)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.title("Spectra at x,z = center")
    plt.show()

# ---------------------------------
def plot_slice(models, molecules):

    if not isinstance(models, list):
        models = [models]

    nmod = len(models)
    fig, axs = plt.subplots(1, nmod, figsize=(5*nmod, 8))

    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    for i, model in enumerate(models):
        step = model.shape[3]//16

        Xo, Yo, Zo, vx, vy, vz, ngas = model[0, :, :, ::step, :]
        Xo1, Yo1, Zo1, vx1, vy1, vz1, ngas1 = model[1, :, :, ::step, :]
        vbulk_pseudo = np.mean(vy)
        vy -= vbulk_pseudo
        vy1 -= vbulk_pseudo

        zcut = ngas.shape[2]//2

        ax = axs[i]

        ax.pcolor(Xo[:, :, zcut], Yo[:, :, zcut], ngas[:, :, zcut], cmap='Blues')
        #plt.colorbar(label=f"n({molecules[0]})")
        ax.quiver(Xo[:, :, zcut], Yo[:, :, zcut], vx[:, :, zcut], vy[:, :, zcut], color='tab:blue', label=molecules[0])
        ax.quiver(Xo1[:, :, zcut], Yo1[:, :, zcut], vx1[:, :, zcut], vy1[:, :, zcut], color='tab:orange', label=molecules[1])
        ax.axvline(0, color='k', linestyle='--', label='LOS', alpha=0.5)
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        ax.set_xlabel("X")

    axs[0].set_ylabel("Y (LOS)")
    for i in range(1, nmod):
        axs[i].set_yticks([])

    plt.tight_layout()
    plt.show()


# ---------------------------------
def plot_comparison(ppvs_target, ppvs_optimized, vchans, molecules, nchans=30):

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
    plt.show()
