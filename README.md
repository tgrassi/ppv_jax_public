# 🌀 PPV optimization JAX/Optax tool
(aka Vectorama)

A **differentiable** 3D geometrical model that produces synthetic PPV data cubes from the parameterized density and velocity fields, and that can be efficiently optimized to reproduce real PPV data cubes.

Paper: T.Grassi et al. (2026) [arXiv:2603.06791](https://arxiv.org/abs/2603.06791)

------------------------------
### 📔 Example Notebooks

- [Complete example with many plots](test.ipynb) (paper Appendix A)   
- [Quick example using mock FITS data](quick.ipynb)     

------------------------------
### 💻 If you want to run it on your machine
It is recommended to use an Nvidia GPU card.     
```
python -m venv env
source env/bin/activate
pip install --upgrade "jax[cuda12]"  # assuming you have CUDA 12
pip install optax
pip install matplotlib
pip install tqdm
```

------------------------------
### 📝 If you want to customize the model
Change the function `f` in [this file](vectorama/model.py).       
The input consists of a dictionary of parameters (`pp`) and a dictionary of spectral data (`pa`).      
The output must be a tuple (`return ppv, em, jnp.stack([Xo, Yo, Zo, vx, vy, vz, ngas])`)
```
    - ppv (ndarray): Position-position-velocity cube (x, z, v) of shape (nx, ny, nv)
    - em (ndarray): Emission cube before line-of-sight integration (x, y, z, v) of shape (nx, ny, nz, nv)
    - stack (ndarray): Model grid with coordinates and velocity components (x, y, z, vx, vy, vz, ngas),
                       where each element has shape (nx, ny, nv).
```

------------------------------
### 🔧 Tested on 
- NVIDIA GTX 1050Ri, 4GB
- NVIDIA Quadro RTX 4000, 8GB
- NVIDIA A100 PCIe, 80GB

------------------------------
### 📣 Cite    
```
@ARTICLE{Grassi2026,
       author = {{Grassi}, T. and {Pineda}, J.~E. and {Spezzano}, S. and {Arzoumanian}, D. and {Lique}, F. and {Misugi}, Y. and {Redaelli}, E. and {Jensen}, S.~S. and {Caselli}, P.},
        title = "{A differentiable and optimizable 3D model for interpretation of observed spectral data cubes}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics of Galaxies, Instrumentation and Methods for Astrophysics},
         year = 2026,
        month = mar,
          eid = {arXiv:2603.06791},
        pages = {arXiv:2603.06791},
          doi = {10.48550/arXiv.2603.06791},
archivePrefix = {arXiv},
       eprint = {2603.06791},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260306791G},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

