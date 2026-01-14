# 🌀 PPV optimization JAX/Optax tool
(aka Vectorama)

### 📔 Example Notebooks

- [Complete example with many plots](test.ipynb) (Appendix A)   
- [Quick example loading mock FITS data](quick.ipynb)     

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

### 🔧 Tested on 
- NVIDIA GTX 1050Ri 4GB
- NVIDIA Quadro RTX 4000 8GB
- NVIDIA A100 80GB PCIe
