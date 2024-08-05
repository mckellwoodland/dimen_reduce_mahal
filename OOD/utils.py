"""
Functions used for the OOD detection evaluation.
"""
# Imports
import os
import torch
import tqdm

import numpy as np
import pandas as pd

from sklearn import metrics

# Functions
def evaluate(in_dist, out_dist):
    """
    Calculate the AUROC and AUPR for given Mahalanobis distances.
    Inputs:
      in_dist (pd series): Pandas series with OOD scores for the in-distribution data.
      out_dist (pd series): Pandas series with OOD scores for the out-of-distribution data.
    Returns:
        auroc (float): Area under the receiver operating characteristic curve.
        aupr (float): Area under the precision-recall curve.
    """
    # Create the ground truth. 0 ~ ID. 1 ~ OOD.
    truth = [0]*len(in_dist) + [1]*len(out_dist)
    # Combine the distances.
    pred = pd.concat([in_dist, out_dist])
    # Normalize the data between 0 and 1.
    pred = (pred - pred.min())/(pred.max()-pred.min())
    # Calculate the fprs and tprs.
    fpr, tpr, thresholds = metrics.roc_curve(truth, pred)
    # Calculate the AUROC.
    auroc = metrics.auc(fpr, tpr)
    # Calculate the AUPR.
    auprc = metrics.average_precision_score(truth, pred)
    # Calculate FPR90.
    fpr90 = get_fpr(in_dist, out_dist, 0.90)
    return round(auroc,2), round(auprc,2), round(fpr90,2)

def get_fpr(in_dist, out_dist, percent):
    """
    Calculate the false positive rate at a give true positive rate.
    Inputs:
        in_dist (pd series): Pandas Series with OOD scores for the in-distribution data.
        out_dist (pd series): Pandas Series with OOD scores for the out-of-distribution data.
        percent (float): True positive rate.
    Returns:
        fpr (float): False positive rate.
    """
    q = out_dist.quantile(q=(1-percent))
    tp = sum(out_dist >= q)
    fp = sum(in_dist >= q)
    tn = sum(in_dist < q)
    fn = sum(out_dist < q)
    fpr = fp / (fp + tn)
    return fpr

def read_embeddings(folder, embed_type, is_numpy, sliding_window):
    """
    Read in bottleneck features.

    Inputs:
        path (str): Path to the folder that has the encodings.
                    PyTorch encodings must be saved as .pt files.
                    NumPy encodings must be saved as .npy files.
        embed_type (str): Whether it is a torch or numpy embedding.
        is_numpy (bool): Whether to return the embedding as a NumPy array.
                         Required for PCA, t-SNE, and UMAP.
        sliding_window (str): Whether to reduce the sliding window dimension with average pooling (avg) or max pooling (max).
                              Neither of these values will result in no reduction.
    Returns:
        (array): all encodings
    """
    embeds = []
    filenames = []
    for pt_file in tqdm.tqdm(os.listdir(folder)):
        pt_path = os.path.join(folder, pt_file)
        if embed_type == "torch":
            encoding = torch.load(pt_path).squeeze()
        elif embed_type == "numpy":
            encoding = np.load(pt_path)
        if sliding_window == "avg":
            encoding = np.mean(encoding, axis=0)
        elif sliding_window == "max":
            encoding = np.max(encoding, axis=0)
        if is_numpy:
            embeds.append(encoding.flatten())
        else:
            if embed_type == "torch":
                embeds.append(encoding)
            else:
                embeds.append(torch.from_numpy(encoding))
        filenames.append(pt_file)
    if is_numpy:
        return np.stack(embeds, axis=0), filenames
    else:
        return torch.stack(embeds, axis=0), filenames
