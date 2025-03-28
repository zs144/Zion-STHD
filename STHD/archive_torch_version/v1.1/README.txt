Luanch date: Mar 27, 2025
Author: Zion Sheng

This version improved the usage of the spatail connectivity matrix, a sparse matrix
that specifies the neighboring spots relationship (i.e., the two spots are neighbours
or not). Previously, this matrix was used in the way to improve the computation
efficiency in Numpy and Numba. Now, there is no need for that, so I rewrote the
related code to align with the practice of PyTorch.

Specifically, the spatail connectivity matrix will be directly used to compute
CE loss, although we need to convert the type first (froms scipy csr_matrix to
tensor sparse_csr_tensor/sparse_coo_tensor matrix). So far, it doesn't really bring
any significant efficiency improvemnt during the testing (on the tutorial patch),
but it makes the code look cleaner, espeicallly the `compute_ce_pytorch()`
function.
