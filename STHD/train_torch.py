"""
Example
python3 STHD/train.py --refile ../testdata/crc_average_expr_genenorm_lambda_98ct_4618gs.txt --patch_list ../testdata/crop10large//patches/52979_9480 ../testdata/crop10large//patches/57479_9480 ../testdata/crop10large//patches/52979_7980 ../testdata/crop10large//patches/55979_7980 ../testdata/crop10large//patches/57479_7980 ../testdata/crop10large//patches/54479_9480 ../testdata/crop10large//patches/55979_9480 ../testdata/crop10large//patches/54479_7980
"""

import argparse
import os
from time import time

import numpy as np
import pandas as pd
from STHD import model, qcmask, refscrna, sthdio

import torch
import torch.nn as nn
import torch.optim as optim

def calculate_ll_pytorch(P, F, X):
    """ Calculate the Poisson log-likelihood loss.
    Parameters
    ----------
    P : torch.tensor
        probability tensor for cell type assignment.
    F : torch.tensor
      $$\sum_g (-lambda + n^g_a * log(-lambda))$$.
    X : int
        number of spots (each spot: a).
    """
    # Element-wise multiplication between P and F
    res = torch.sum(P * F)
    res = res / X
    return res


def csr_obtain_column_index_for_row(row, column, i):
    # get row i's non-zero items' column index
    row_start = row[i]
    row_end = row[i + 1]
    column_indices = column[row_start:row_end]
    return column_indices


def calculate_ce_pytorch(P, A, X):
    """ Calculate the cross-entropy loss for neighborhood similarity.
    Parameters
    ----------
    P : torch.tensor
        probability tensor for cell type assignment.
    Ascr_row : list
        indptr for (CSR) sparse matrix representation of space connectivity.
    Acsr_col : list
        indices for (CSR) sparse matrix representation of space connectivity.
    X : int
        number of spots (each spot: a).
    """
    # Compute G[a,t] = Σ_{a_star in neighbors(a)} log(P[a_star,t])
    logP = torch.log(P)
    G = torch.sparse.mm(A, logP)

    # Element-wise multiplication between P and G
    res = - torch.sum(P * G)
    res = res / X
    return res


def scipy_csr_to_torch_csr(scipy_csr_matrix, device):
    crow_indices = torch.from_numpy(scipy_csr_matrix.indptr).long()
    col_indices = torch.from_numpy(scipy_csr_matrix.indices).long()
    data_values = torch.from_numpy(scipy_csr_matrix.data)
    size = scipy_csr_matrix.shape

    torch_csr_matrix = torch.sparse_csr_tensor(
        crow_indices, col_indices, data_values, size=size,
        device=device, dtype=torch.float32
    )
    return torch_csr_matrix


def scipy_coo_to_torch_coo(scipy_coo_matrix, device):
    row_indices = scipy_coo_matrix.row
    col_indices = scipy_coo_matrix.col
    indices = torch.LongTensor(np.array([row_indices, col_indices]))
    data_values = torch.from_numpy(scipy_coo_matrix.data)
    size = scipy_coo_matrix.shape

    torch_coo_matrix = torch.sparse_coo_tensor(
        indices, data_values, size=size, device=device, dtype=torch.float32
    )
    return torch_coo_matrix


def train_pytorch(sthd_data, n_iter, step_size, beta,
                  device='cuda' if torch.cuda.is_available() else 'cpu'):
    print("[Log] Preparing constants and training weights")

    # Prepare constants
    # - X: number of spots (each spot: a).
    # - Y: number of genes after filtering (each gene: g).
    # - Z: number of cell types (each type: t).
    # - F: $$ \sum_g (-lambda + n^g_a * log(-lambda)) $$. shape=(X, Z).
    #      The part of LL loss without the parameters, so it can be precomputed.
    # - A: CSR sparse matrix representation of space connectivity.
    X, Y, Z, F, A = model.prepare_constants_torch(sthd_data)
    # A = scipy_csr_to_torch_csr(A, device=device)
    A = A.tocoo()
    A = scipy_coo_to_torch_coo(A, device=device)

    # Convert to PyTorch tensors and move to device
    F = torch.tensor(F, dtype=torch.float32, device=device)

    # Initialize model parameters (weights)
    # W: weight matrix of each cell type at each spot.
    W = torch.ones((X, Z), device=device, dtype=torch.float32, requires_grad=True)
    # P: probability tensor for cell type assignment.
    # $$ P_a(t) = softmax(W) = exp(w_a^t) / sum(exp(w_a^t)) $$
    # dim=1 so that the sum of each row (spot) is 1.
    P = torch.softmax(W, dim=1, dtype=torch.float32)

    # Optimizer
    optimizer = optim.Adam([W], lr=step_size)

    # Loss functions
    print("[Log] Training...")
    print(f"{'iter':<10}{'time (min)':<15}{'total loss':<15}{'LL loss':<15}{'CE loss':<15}")
    for i in range(n_iter):
        start = time()
        optimizer.zero_grad()

        # Poisson log-likelihood loss for gene expression modeling
        ll_loss = calculate_ll_pytorch(P, F, X)

        # Cross-entropy loss for neighborhood similarity
        ce_loss = calculate_ce_pytorch(P, A, X)

        # Total loss (weighted sum)
        loss = -ll_loss + beta * ce_loss

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Update probability tensor
        P = torch.softmax(W, dim=1, dtype=torch.float32)

        end = time()
        duration = (end - start) / 60.0
        print(f'{i:<10}{duration:<15.2f}{loss.detach().cpu():<15.4f}' +
              f'{ll_loss.detach().cpu():<15.4f}{ce_loss.detach().cpu():<15.4f}')

    print("[Log] Training complete.")
    return P  # Return final cell type probabilities


def sthdata_match_refgene(sthd_data, refile, ref_gene_filter=True, ref_renorm=False):
    genemeanpd_filtered = refscrna.load_scrna_ref(refile)
    ng1 = sthd_data.adata.shape[1]
    sthd_data.match_refscrna(genemeanpd_filtered, cutgene=True, ref_renorm=ref_renorm)
    ng2 = sthd_data.adata.shape[1]
    print(
        f"cut {ng1} genes to match to reference {ng2} genes",
    )
    if ref_gene_filter:
        genemeanpd_filtered = genemeanpd_filtered.loc[sthd_data.adata.var_names]
    return (sthd_data, genemeanpd_filtered)


def train(sthd_data, n_iter, step_size, beta, debug=False, early_stop=False):
    print("[Log] prepare_constants and training weights")
    X, Y, Z, F, Acsr_row, Acsr_col = model.prepare_constants(sthd_data)
    W, eW, P, Phi, ll_wat, ce_wat, m, v = model.prepare_training_weights(X, Y, Z)
    print("[Log] Training...")
    metrics = model.train(
        n_iter=n_iter,
        step_size=step_size,  # learnin rate
        beta=beta,  # weight of CE w.r.t log-likelihood
        constants=(X, Y, Z, F, Acsr_row, Acsr_col),
        weights=(W, eW, P, Phi, ll_wat, ce_wat, m, v),
        early_stop=early_stop,  # True will trigger early_stop criteria, False will run through all n_iter
    )
    print("[Log] Training complete.")
    if debug:
        return P, metrics
    else:
        return P


def fill_p_filtered_to_p_full(
    P_filtered, sthd_data_filtered, genemeanpd_filtered, sthd_data
):
    """For prediction performed for filtered data, put P_filtered back to full data size and fill -1.
    sthd_data cannot already have the probability columns
    """
    P_filtered_df = pd.DataFrame(
        P_filtered,
        index=sthd_data_filtered.adata.obs.index,
        columns=genemeanpd_filtered.columns,
    )
    P_filtered_df.columns = "p_ct_" + P_filtered_df.columns
    p_columns = P_filtered_df.columns

    obs_withp = sthd_data.adata.obs.merge(
        P_filtered_df, how="left", left_index=True, right_index=True
    )
    P = obs_withp[list(p_columns)].fillna(-1).values
    return P


def predict(sthd_data, p, genemeanpd_filtered, mapcut=0.8):
    """Based on per barcode per cell type probability, predict cell type and put prediction in adata.
    sthd_data = predict(sthd_data, p, genemeanpd_filtered, mapcut= 0.8)
    """
    adata = sthd_data.adata.copy()
    for i, ct in zip(
        range(len(genemeanpd_filtered.columns)), genemeanpd_filtered.columns
    ):
        adata.obs["p_ct_" + ct] = p[:, i]
    adata.obs["x"] = adata.obsm["spatial"][:, 0]
    adata.obs["y"] = adata.obsm["spatial"][:, 1]

    # get map predictions
    STHD_prob = adata.obs[[t for t in adata.obs.columns if "p_ct_" in t]]
    ct_max = STHD_prob.columns[STHD_prob.values.argmax(1)]
    STHD_pred_ct = pd.DataFrame({"ct_max": ct_max}, index=STHD_prob.index)
    STHD_pred_ct["ct"] = STHD_pred_ct["ct_max"]

    # assign ambiguous based on posterior cut
    ambiguous_mask = (STHD_prob.max(axis=1) < mapcut).values
    STHD_pred_ct.loc[ambiguous_mask, "ct"] = "ambiguous"

    # assign filtered region to 'filtered'
    filtered_mask = (
        STHD_prob.max(1) < 0
    ).values  # by default, filtered spots have prob as -1.
    STHD_pred_ct.loc[filtered_mask, "ct"] = "filtered"

    # assign final cell type prediction
    adata.obs["STHD_pred_ct"] = STHD_pred_ct["ct"]
    print("[Log]Predicted cell type in STHD_pred_ct in adata.obs")
    print(
        "[Log]Predicted cell type probabilities in columns starting with p_ct_ in adata.obs"
    )
    sthd_data.adata = adata

    return sthd_data


########## Training IO: Saving


def save_prediction_pdata(sthdata, file_path=None, prefix=""):
    """Save from sthdata the pdata into dataframe with probabilities, predicted cell type, and x, y

    Example:
    -------
    pdata = train.save_prediction_pdata(sthdata, file_path = '', prefix = '')

    """
    predcols = (
        ["x", "y"]
        + ["STHD_pred_ct"]
        + [t for t in sthdata.adata.obs.columns if "p_ct_" in t]
    )
    pdata = sthdata.adata.obs[predcols]

    if file_path is not None:
        pdata_path = os.path.join(file_path, prefix + "_pdata.tsv")
        pdata.to_csv(pdata_path, sep="\t")
        print(f"[Log] prediction saved to {pdata_path}")
    return pdata


########## Training IO: Loading


def load_data(file_path):
    """Load expr data. Only works with cropped data."""
    sthd_data = sthdio.STHD(
        spatial_path=os.path.join(file_path, "adata.h5ad.gzip"),
        counts_data=None,
        full_res_image_path=os.path.join(file_path, "fullresimg_path.json"),
        load_type="crop",
    )
    print("[log] Number of spots: ", sthd_data.adata.shape[0])
    return sthd_data


def load_pdata(file_path, prefix=""):
    pdata_path = os.path.join(file_path, prefix + "_pdata.tsv")
    pdata = pd.read_table(pdata_path, index_col=0)
    return pdata


def add_pdata(sthd_data, pdata):
    """Load from pdata into dataframe with probabilities, predicted cell type, and x, y, and put in sthdata
    to rename: put_pdata_to_sthdata

    Example:
    -------
    pdata = train.add_pdata(sthdata, pdata)

    """
    sthdata = sthd_data
    exist_cols = sthdata.adata.obs.columns.intersection(pdata.columns)
    print("[Log] Loading prediction into sthdata.adata.obs, overwriting")
    print(exist_cols)
    for col in sthdata.adata.obs[exist_cols]:
        del sthdata.adata.obs[col]
    sthdata.adata.obs = sthdata.adata.obs.merge(
        pdata, how="left", left_index=True, right_index=True
    )
    return sthdata


def load_data_with_pdata(file_path, pdata_prefix=""):
    """A simplified
    Load full prediction for the patch. need os.path.join(file_path, pdata_prefix+_pdata.h5ad)
    Return adata with probability and predicted cell type in .obs
    contains all gene expression
    [] todo: deprecate
    """
    sthdata = load_data(file_path)
    pdata = load_pdata(file_path, pdata_prefix)
    sthdata_with_pdata = add_pdata(sthdata, pdata)
    return sthdata_with_pdata


##########


def main(args):
    start = time()
    for patch_path in args.patch_list:
        print(f"[log] {time() - start:.2f}, start processing patch {patch_path}")
        if args.filtermask:
            sthdata = load_data(patch_path)
            print(sthdata.adata.shape)
            sthdata.adata = qcmask.background_detector(
                sthdata.adata,
                threshold=args.filtermask_threshold,
                n_neighs=4,
                n_rings=args.filtermask_nrings,
            )
            print(sthdata.adata.shape)
            # visualize_background(sthdata)
            sthdata_filtered = qcmask.filter_background(
                sthdata, threshold=args.filtermask_threshold
            )
            sthdata_filtered, genemeanpd_filtered = sthdata_match_refgene(
                sthdata_filtered, args.refile, ref_renorm=args.ref_renorm
            )
            print("[Log]Training")
            P_filtered = train(sthdata_filtered, args.n_iter, args.step_size, args.beta)
            P = fill_p_filtered_to_p_full(
                P_filtered, sthdata_filtered, genemeanpd_filtered, sthdata
            )
        else:
            sthdata = load_data(patch_path)
            sthdata, genemeanpd_filtered = sthdata_match_refgene(
                sthdata, args.refile, ref_renorm=args.ref_renorm
            )
            print("[Log]Training")
            P = train(sthdata, args.n_iter, args.step_size, args.beta)

        sthdata = predict(sthdata, P, genemeanpd_filtered, mapcut=args.mapcut)
        _ = save_prediction_pdata(sthdata, file_path=patch_path, prefix="")
        print("[Log]prediction saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_iter",
        default=23,
        type=int,
        help="iteration to optimize LL and CE. recommended 23 (tuned for human colon cancer sample)0",
    )
    parser.add_argument("--step_size", default=1, type=int)
    parser.add_argument(
        "--beta",
        default=0.1,
        type=float,
        help="beta parameter for borrowing neighbor info. recommended 0.1",
    )
    parser.add_argument(
        "--mapcut",
        default=0.8,
        type=float,
        help="posterior cutoff for celltype prediction",
    )
    parser.add_argument(
        "--refile", type=str, help="reference normalized gene mean expression."
    )
    parser.add_argument(
        "--ref_renorm", default=False, type=bool, help="recommended False"
    )
    parser.add_argument(
        "--filtermask", type=bool, default=True, help="whether to filter masked spots"
    )
    parser.add_argument(
        "--filtermask_nrings",
        default=2,
        type=int,
        help="auto detection of low-count region: number of rings to consider neighbor",
    )
    parser.add_argument(
        "--filtermask_threshold",
        default=51,
        type=int,
        help="auto detection of low-count region: number of total counts",
    )
    parser.add_argument(
        "--patch_list", nargs="+", default=[], help="a space separated patch path list"
    )

    args = parser.parse_args()

    main(args)

    """
    # quick test
    class Args:
    def __init__(self):
        self.n_iter=10
        self.step_size = 1
        self.beta = 0.1
        self.refile =  '../testdata/crc_average_expr_genenorm_lambda_98ct_4618gs.txt'
        self.filtermask = True
        self.patch_list = ['../testdata/crop10']
    train.main(Args())
    """
