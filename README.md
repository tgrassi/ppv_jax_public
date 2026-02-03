# 🌀 PPV optimization JAX/Optax tool
(aka Vectorama)

Paper [submitted](https://www.aanda.org/)

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
(Paper still under review, soon on arXiv)

