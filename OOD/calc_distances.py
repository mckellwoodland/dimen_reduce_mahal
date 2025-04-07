"""
Calculate the distance between testing embeddings and a distribution of training embeddings.
"""
# Imports.
import argparse
import os
import time
import torch
import tqdm
import umap

import nibabel as nib
import numpy as np
import pandas as pd

from os import path
from scipy import linalg
from scipy.spatial import distance
from sklearn import decomposition, manifold, preprocessing
from torch import nn

# Arguments.
parser = argparse.ArgumentParser()
parser._action_groups.pop()

def none_or_val(value):
    if value == "None":
        return None
    elif value.isdigit():
        return int(value)
    return value

required = parser.add_argument_group('Required Arguments')
required.add_argument('-tr', '--train_dir', type=str, required=True, help='Path to directory containing the train image features.')
required.add_argument('-te', '--test_dir', type=str, required=True, help='Path to directory containing the test image features.')
required.add_argument('-o', '--out_dir', type=str, required=True, help='Path to directory to put distances into.')

optional = parser.add_argument_group('Optional Arguments')
optional.add_argument('-d', '--distance', type=none_or_val, default=None, help='Distance: Mahalanobis distance [md] or k-th nearest neighbor distance [knn]. \
                                                                                If None, a hyperparameter search will be performed. \
                                                                                Defaults to knn.')
optional.add_argument('-e', '--embed_type', type=str, default='numpy', help='Whether the embeddings were saved as Torch embddings in .pt files [torch], \
                                                                             NumPy embeddings in .npy files [numpy], or \
                                                                             NIfTI embeddings in .nii.gz files [nifti].')
optional.add_argument('-r', '--reduce_type', type=none_or_val, default=None, help='Dimensionality reduction technique: PCA [pca], t-SNE [tsne], UMAP [umap],\
                                                                                   or average pooling [avgpool]. \
                                                                                   If None, a hyperparameter search will be performed. \
                                                                                   Defaults to None.')
optional.add_argument('-s', '--sliding_window', type=none_or_val, default=None, help='Whether to reduce the sliding window dimension \
                                                                                      with average pooling [avgpool] or max pooling [maxpool]. \
                                                                                      Defaults to no reduction [None].')

k_opt = parser.add_argument_group('K-th Nearest Neighbor Distance')
k_opt.add_argument('-k', type=none_or_val, default=None, help='k in the k-th nearest neighbor distance. \
                                                              If None, a hyperparameter search will be performed. \
                                                              Defaults to None.')

avg_pool_opt = parser.add_argument_group('Average Pooling Arguments')
avg_pool_opt.add_argument('-di', '--dim', type=none_or_val, default=None, help='2- or 3-dimensional reduction [2D, 3D]. \
                                                                                If None, a hyperparameter search will be performed. \
                                                                                Defaults to None.')
avg_pool_opt.add_argument('-ke', '--kernel_size', type=none_or_val, default=None, help='Kernel size. \
                                                                                        If None, a search will be performed over kernel size and stride. \
                                                                                        Defaults to None.')
avg_pool_opt.add_argument('-st', '--stride', type=none_or_val, default=None, help='Stride. \
                                                                                   If None, a search will be performed over kernel size and stride. \
                                                                                   Defaults to None.')

pca_opt = parser.add_argument_group('PCA/UMAP/t-SNE Argument')
pca_opt.add_argument('-n', '--num_comp', type=none_or_val, default=None, help='Number of components. \
                                                                               If None, a hyperparameter search will be performed. \
                                                                               Defaults to None.')

args = parser.parse_args()

if not path.exists(args.out_dir):
    os.mkdir(args.out_dir)

# Classes.
class AvgPool(nn.Module):
    def __init__(self, kernel_size, stride, dim):
        super(AvgPool, self).__init__()
        if dim == "2D":
            self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        elif dim == "3D":
            self.avgpool = nn.AvgPool3d(kernel_size=kernel_size, stride=stride)
    def forward(self, x):
        x = self.avgpool(x)
        return x

# Functions.
def calc_dist(train, test, test_f, distance, k, max_n, reduce_str, reduce_time, seconds, strings):
    """
    Evaluate the k-th nearest neighbor distance for a specified k and dimensionality-reduction setting.

    Inputs:
        train (list): Train embeddings.
        test (list): Test embeddings.
        test_f (list): Test filenames.
        distance (string): Mahalanobis [md], k-th nearest neighbor [knn], or hyperparameter search [None].
        k (int): Hyperparameter k.
                 If None and distance is knn, a hyperparameter search will be performed.
        max_n (int): 2**n is the largest parameter for k that will be tested in a hyperparameter search.
        reduce_str (str): String containing information about the reduciton for the purpose of the output file name.
        reduce_time (float): The amount of time it took to reduce the dimensionality of the embeddings.
        seconds (list): The amount of time it took to get distances (plus reduction).
        strings (list): The reduction strings.
    """
    # K-th nearest neighbor.
    if distance == 'knn' or not distance:
        if k:
            ks = [k]
        else:
            # Hyperparameter search.
            ks = [2**n for n in range(1,max_n+1)]
        for k in ks:
            # Calculate distances.
            start = time.time()
            test_dists = kth_distance(test, train, k, test_f)

            # Save results.
            seconds.append(time.time() - start)
            strings.append(f'knn_k-{k}_r-{reduce_str}')
            test_dists.to_csv(path.join(args.out_dir, f'test_distances_knn_k-{k}_r-{reduce_str}.csv'), index=False)

    # Mahalanobis distance.
    if distance == 'md' or not distance:
        # Calculate distances.
        start = time.time()
        mean, inv_cov = get_stats(train)
        test_dists = pd.DataFrame({'filename':[f.split('.')[0] for f in test_f],
                                     'dists': [mahal_dist(x, mean, inv_cov) for x in test]})

        # Save results.
        seconds.append(time.time() - start)
        strings.append(f'md_r-{reduce_str}')
        test_dists.to_csv(path.join(args.out_dir, f'test_distances_md_r-{reduce_str}.csv'), index=False)
    return seconds, strings

def get_stats(embed):
    """
    Calculate the mean and the inverse of the covariance matrix in preparation for the Mahalanobis distance calculation.

    Input:
        embed (array): Embeddings to fit the Gaussian distribution to.
    
    Output:
        mean (ndarray): Mean of the distribution.
        inv_cov (ndarray): Inverse of the covariance matrix of the distribution.
    """
    mean = np.mean(embed, axis=0, keepdims=True).squeeze()
    cov = np.cov(embed.T)
    inv_cov = linalg.inv(cov)
    return mean, inv_cov

def kth_distance(train, test, k, test_f):
    """
    Calculate the k-th nearest neighbor distance.

    Input:
        train (list): Train embeddings.
        test (list): Test embeddings.
        k (int): Hyperparameter k.
        test_f (list): Test files.

    Return:
        Pandas dataframe with distances.
    """
    dists = distance.cdist(test, train)
    return pd.DataFrame({'filename':[f.split('.')[0] for f in test_f],
                         'dist':[sorted(row)[k] for row in dists]})

def mahal_dist(x, mean, inv_cov):
    """
    Calculate the Mahalanobis distance between point x and a distribution
    with mean mean and covariance cov.

    Inputs:
        x (ndarray): new point
        mean (ndarray): mean of the distribution
        cov (ndarray): inverse of the covariance matrix for the distribution
    
    Returns:
        (float): Mahalanobis distance
    """
    diff = x - mean
    d_squared = np.matmul(np.matmul(diff, inv_cov), diff)
    return np.sqrt(np.abs(d_squared))

def read_embeddings(folder, embed_type, is_numpy, sliding_window):
    """
    Read in bottleneck features.

    Inputs:
        folder (str): Path to the folder that has the encodings.
                      PyTorch encodings must be saved as .pt files.
                      NumPy encodings must be saved as .npy files.
                      NIfTI encodings must be saved as .nii.gz files.
        embed_type (str): Whether embeddings were saved as Torch, NumPy, or NIfTI files.
        is_numpy (bool): Whether to return the embedding as a NumPy array.
                         Required for PCA, t-SNE, and UMAP.
        sliding_window (str): Whether to reduce the sliding window dimension with average pooling (avg) or max pooling (max).
                              Neither of these values will result in no reduction.
    
    Returns:
        (array): all encodings
    """
    embeds = []
    filenames = []
    print("Reading in embeddings")
    for embed_file in tqdm.tqdm(os.listdir(folder)):
        embed_path = os.path.join(folder, embed_file)
        file_end = '.'.join(embed_path.split('.')[1:])
        if file_end == "pt" and embed_type == "torch":
            embedding = torch.load(embed_path).squeeze()
        elif file_end == "npy" and embed_type == "numpy":
            embedding = np.load(embed_path).squeeze()
        elif file_end == "nii.gz" and embed_type == "nifti":
            embedding = nib.load(embed_path).get_fdata().squeeze()
        else:
            continue
        if sliding_window == "avg":
            embedding = np.mean(embedding, axis=0)
        elif sliding_window == "max":
            embedding = np.max(embedding, axis=0)
        if is_numpy:
            embeds.append(embedding.flatten())
        else:
            if embed_type == "torch":
                embeds.append(embedding)
            else:
                embeds.append(torch.from_numpy(embedding))
        filenames.append(embed_file)
    if is_numpy:
        return np.stack(embeds, axis=0), filenames
    else:
        return torch.stack(embeds, axis=0), filenames

def reduce_avgpool(dataframe, kernel_size, stride, dim):
    """
    Perform average pooling on embedding to reduce dimensionality.
    
    Inputs:
        dataframe (array): Embeddings to be pooled.
        kernel_size (int): Size of the pooling kernel.
        stride (int): Stride of the pooling kernel.
        dim (str): either '2D' or '3D' for dimensionality of the embedding.
    
    Returns:
        Baseline embeddings with dimensionality reduced by PCA.
        Personalization embeddings with dimensionality reduced by PCA.
    """
    model = AvgPool(kernel_size, stride, dim)
    return np.array([model(x).numpy().flatten() for x in dataframe])

def reduce_pca(train, test, num_comp):
    """
    Reduce the dimensionality of the embeddings with Principal Component Analysis (PCA).
    
    Inputs:
        train (array): Train embeddings.
        test (array): Test embeddings.
        num_comp (int): The number of components for PCA.
    
    Outputs:
        train (array): Reduced train embeddings.
        test (array): Reduced test embeddings.
    """
    scale = preprocessing.StandardScaler()
    train = scale.fit_transform(train)
    test = scale.transform(test)

    pca = decomposition.PCA(n_components=num_comp)
    train = pca.fit_transform(train)
    test = pca.transform(test)
    return train, test

def reduce_tsne(train, test, num_comp):
    """
    Reduce the dimensionality of the embeddings with t-Distributed Stochastic Neighbor Embedding (t-SNE).

    Inputs:
        train (array): Train embeddings.
        test (array): Test embeddings.
        num_comp (int): The number of components for t-SNE.

    Outputs:
        Reduced train embeddings.
        Reduced test embeddings.
    """
    combine = np.vstack([train, test])
    model = manifold.TSNE(num_comp)
    combine_t = model.fit_transform(combine)
    return combine_t[:len(train)], combine_t[len(train):]

def reduce_umap(train, test, num_comp):
    """
    Reduce the dimensionality of the embeddings with Uniform Manifold Approximation and Projection (UMAP).

    Inputs:
        train (array): Train embeddings.
        test (array): Test embeddings.
        num_comp (int): The number of oomponents.
    Outputs:
        Reduced train embeddings.
        Reduced test embeddings.
    """
    combine = np.vstack([train, test])
    model = umap.UMAP(n_components=num_comp)
    combine_t = model.fit_transform(combine)
    train_start, train_end = 0, len(train)
    return combine_t[:len(train)], combine_t[len(train):]

# Main script.
if __name__=="__main__":
    # Read in embeddings.
    if not args.reduce_type == 'avgpool':
        # NumPy embeddings.
        train, train_f = read_embeddings(args.train_dir, embed_type=args.embed_type, is_numpy=True, sliding_window=args.sliding_window)
        test, test_f = read_embeddings(args.test_dir, embed_type=args.embed_type, is_numpy=True, sliding_window=args.sliding_window)

    # Calculate maximum number of components/neighbors.
    num_train = len(train)
    max_n = 1
    while 2**(max_n+1) < num_train:
        max_n += 1

    # Keep track of time.
    seconds, strings = [], []

    # PCA reduction.
    if args.reduce_type == 'pca' or not args.reduce_type:
        print("PCA Reduction")
        if args.num_comp:
            num_comps = [args.num_comp]
        else:
            num_comps = [2**n for n in range(1, max_n+1)]
        for num_comp in num_comps:
            start = time.time()
            train_reduce, test_reduce = reduce_pca(train, test, num_comp)
            seconds, strings = calc_dist(train_reduce, test_reduce, test_f, args.distance, args.k, max_n, f'pca_n-{num_comp}', time.time()-start, seconds, strings)

    # t-SNE reduction.
    if args.reduce_type == 'tsne' or not args.reduce_type:
        print("t-SNE Reduction")
        start = time.time()
        train_reduce, test_reduce = reduce_tsne(train, test, 2)
        seconds, strings = calc_dist(train_reduce, test_reduce, test_f, args.distance, args.k, max_n, 'tsne', time.time()-start, seconds, strings)
    
    # UMAP.
    if args.reduce_type == 'umap' or not args.reduce_type:
        print("UMAP Reduction")
        if args.num_comp:
            num_comps = [args.num_comp]
        else:
            # Hyperparameter search.
            num_comps = [2**n for n in range(1, max_n+1)]

        for num_comp in num_comps:
            train_reduce, test_reduce = reduce_umap(train, test, num_comp)
            seconds, strings = calc_dist(train_reduce, test_reduce, test_f, args.distance, args.k, max_n, f'umap_n-{num_comp}', time.time()-start, seconds, strings)

    # Average pooling.
    if args.reduce_type == 'avgpool' or not args.reduce_type:
        print("Average Pooling Reduction")
        # Get Torch embeddings.
        train, train_f = read_embeddings(args.train_dir, embed_type=args.embed_type, is_numpy=False, sliding_window=args.sliding_window)
        test, test_f = read_embeddings(args.test_dir, embed_type=args.embed_type, is_numpy=False, sliding_window=args.sliding_window)
         
        if args.dim:
            dims = [args.dim]
        else:
            #Hyperparameter sarch
            dims = ['2D','3D']

        if args.kernel_size and args.stride:
            kern_str = [(args.kernel_size, args.stride)]
        else:
            # Hyperparameter search
            kern_str = [(2,1),(2,2),(3,1),(3,2),(4,1)]

        for dim in dims:
            for kernel, stride in kern_str:
                start = time.time()
                train_reduce = reduce_avgpool(train, kernel, stride, dim)
                test_reduce = reduce_avgpool(test, kernel, stride, dim)
                seconds, strings = calc_dist(train_reduce, test_reduce, test_f, args.distance, args.k, max_n, f'avgpool_k-{kernel}_s-{stride}_d-{dim}', time.time()-start, seconds, strings)
    time_df = pd.DataFrame({'reduce_type': strings,
                            'time': seconds})
    time_df.to_csv(path.join(args.out_dir, 'time_taken.csv'), index=False)
