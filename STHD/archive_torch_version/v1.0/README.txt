Luanch date: Mar 26, 2025
Author: Zion Sheng

This is the first working version of the PyTorch-based training algorithm for STHD.
Originally, the training algorithm was implemented using Numpy and Numba. It turns
out that we only need to change a few functions in ./train.py and ./model.py to
move the training into PyTorch and be able to use GPU. Therefore, the code was not
changed too much from the v0.0 (the original code).