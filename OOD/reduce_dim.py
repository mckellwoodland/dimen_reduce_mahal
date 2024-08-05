"""
Reduce the dimensionality of the embeddings using average pooling, PCA, t-SNE, or UMAP.
Encodings must be '.pt' files.
"""
# Imports
import argparse
import os
import torch
import umap
import tqdm
import numpy as np
from sklearn import decomposition, preprocessing, manifold
from torch import nn

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_in', type=str, required='True', help='Path to folder containing the input training embeddings')
parser.add_argument('--train_out', type=str, required='True', help='Path to folder to contain the output training embeddings')
parser.add_argument('--ID_in', type=str, required='True', help='Path to folder containing the input in-distribution test embeddings')
parser.add_argument('--ID_out', type=str, required='True', help='Path to folder to contain the output in-distribution test embeddings')
parser.add_argument('--OOD_in', type=str, required='True', help='Path to folder containing the input out-of-distribution test embeddings')
parser.add_argument('--OOD_out', type=str, required='True', help='Path to folder to contain the output out-of-distribution test embeddings')
parser.add_argument('--type', type=str, required='True', help='Type of dimensionality reduction to use: "avgpool", "pca", "tsne", or "umap"')
parser.add_argument('--num_comp', type=int, help='Number of components to use for PCA, t-SNE, or UMAP')
parser.add_argument('--kernel', type=int, help='Kernel size for average pooling')
parser.add_argument('--stride', type=int, help='Stride size for average pooling')
parser.add_argument('--dim', type=str, help='Dimensionality of pooling for average pooling: "2D" or "3D"')
parser.add_argument('--embed_type', type=str, help='Embedding file type. Options: numpy, torch.')
parser.add_argument('--sliding_window', type=str, help='Reduce sliding window dimension with average (avg) or max (max) pooling.')

args = parser.parse_args()
train_in = args.train_in
ID_in = args.ID_in
OOD_in = args.OOD_in

train_out = args.train_out
ID_out = args.ID_out
OOD_out = args.OOD_out
# Make output directories if they don't exist already.
if not os.path.exists(train_out):
    os.mkdir(train_out)
if not os.path.exists(ID_out):
    os.mkdir(ID_out)
if not os.path.exists(OOD_out):
    os.mkdir(OOD_out)
                      
reduce_type = args.type
assert(reduce_type in ['avgpool', 'pca', 'tsne', 'umap'])
num_comp = args.num_comp
kernel_size = args.kernel
stride = args.stride
dim = args.dim

# Classes
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
    
# Functions
def read_embeddings(folder, embed_type, is_numpy, sliding_window):
    """
    Read in bottleneck features.

    Inputs:
        path (str): Path to the folder that has the encodings.
                    Encodings must be '.pt' files.
        embed_type (str): whether it is a torch or numpy embedding.
        is_numpy (bool): Whether to return the embedding as a NumPy array.
                         Required for PCA, t-SNE, and UMAP.
        sliding_window (str): whether to reduce the sliding window dimension with average pooling (avg) or max pooling (max). Neither of these values will result in no reduction.
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

def reduce_avgpool(train, in_dist, out_dist, kernel_size, stride, dim):
    """
   Perform average pooling on embedding to reduce dimensionality.
    Inputs:
        train (array): training embeddings
        in_dist (array): ID test embeddings
        out_dist (array): OOD test embedings
        kernel_size (int): Size of the pooling kernel
        stride (int): Stride of the pooling kernel
        dim (str): either '2D' or '3D' for dimensionality of the embedding
    Returns:
        train_avgpool (array): training embeddings with dimensionality reduced by PCA
        ID_avgpool (array): in-distribution test embeddings with dimensionality reduced by PCA
        OOD_avgpool (array): OOD test embeddings with dimensionality reduced by PCA
    """
    model = AvgPool(kernel_size, stride, dim)
    train_avgpool = np.array([model(t).numpy().flatten() for t in train])
    ID_avgpool = np.array([model(i).numpy().flatten() for i in in_dist])
    OOD_avgpool = np.array([model(o).numpy().flatten() for o in out_dist])
    return train_avgpool, ID_avgpool, OOD_avgpool
  
def reduce_pca(train, in_dist, out_dist, num_comp):
    """
    Perform PCA on embeddings to reduce dimensionality.
    Inputs:
        train (array): training embeddings
        in_dist (array): ID test embeddings
        out_dist (array): OOD test embedings
        num_comp (int): Number of components to select for PCA
    Returns:
        train_pca (array): training embeddings with dimensionality reduced by PCA
        in_pca (array): in-distribution test embeddings with dimensionality reduced by PCA
        out_pca (array): OOD test embeddings with dimensionality reduced by PCA
    """ 
    # Scale embeddings.
    scaler = preprocessing.StandardScaler()
    train = scaler.fit_transform(train)
    in_dist = scaler.transform(in_dist)
    out_dist = scaler.transform(out_dist)
    
    model = decomposition.PCA(n_components=num_comp)
    train_pca = model.fit_transform(train)
    in_pca = model.transform(in_dist)
    out_pca = model.transform(out_dist)
    return train_pca, in_pca, out_pca
  
def reduce_tsne(train, in_dist, out_dist, num_comp):
    """
    Perform t-SNE on embeddings to reduce dimensionality.
    Inputs:
        train (array): training embeddings
        in_dist (array): ID test embeddings
        out_dist (array): OOD test embedings
        num_comp (int): Number of components
    Returns:
        (array): training embeddings with dimensionality reduced
        (array): ID test embeddings with dimensionality reduced
        (array): OOD test embeddings with dimensionality reduced
    """
    n_t, n_i, n_o = train.shape[0], in_dist.shape[0], out_dist.shape[0]
    combine = np.vstack([train, in_dist, out_dist])
    
    model = manifold.TSNE(num_comp)
    combine_t = model.fit_transform(combine)
    return combine_t[:n_t], combine_t[n_t:n_t+n_i], combine_t[n_t+n_i:]
  
def reduce_umap(train, in_dist, out_dist, num_comp):
    """
    Perform UMAP on embeddings to reduce dimensionality.
    Inputs:
        train (array): training embeddings
        in_dist (array): ID test embeddings
        out_dist (array): OOD test embedings
        num_comp (int): Number of components
    Returns:
        (array): training embeddings with dimensionality reduced
        (array): ID test embeddings with dimensionality reduced
        (array): OOD test embeddings with dimensionality reduced
    """
    n_t, n_i, n_o = train.shape[0], in_dist.shape[0], out_dist.shape[0]
    combine = np.vstack([train, in_dist, out_dist])
    
    model = umap.UMAP(n_components=num_comp)
    combine_u = model.fit_transform(combine)
    return combine_u[:n_t], combine_u[n_t:n_t+n_i], combine_u[n_t+n_i:]
  
def save_embeddings(folder, filenames, embeddings, is_numpy):
    """
    Save off the embeddings whose dimensions have been reduced.
    Inputs:
        folder (str): Path to folder to save the embeddings in
        filenames (list): List of filenames for the embeddings
        embeddings (array): embeddings
        is_numpy (bool): Whether the embedding is a NumPy array.
                         True for PCA, t-SNE, and UMAP.
    """
    for i, pt_file in enumerate(filenames):
        if is_numpy:
            embed = torch.from_numpy(embeddings[i])
        else:
            embed = embeddings[i]
        torch.save(embed, os.path.join(folder, pt_file))
 
# Read in embeddings.
if reduce_type == 'avgpool':
    train, train_files = read_embeddings(train_in, args.embed_type, False, args.sliding_window)
    in_dist, in_files = read_embeddings(ID_in, args.embed_type, False, args.sliding_window)
    out_dist, out_files = read_embeddings(OOD_in, args.embed_type, False, args.sliding_window)
else:
    train, train_files = read_embeddings(train_in, args.embed_type, True, args.sliding_window)
    in_dist, in_files = read_embeddings(ID_in, args.embed_type, True, args.sliding_window)
    out_dist, out_files = read_embeddings(OOD_in, args.embed_type, True, args.sliding_window)

# Reduce embeddings.
if reduce_type == 'avgpool':
  train_r, in_r, out_r = reduce_avgpool(train, in_dist, out_dist, kernel_size, stride, dim)
  print(in_r.shape)
if reduce_type == 'pca':
  train_r, in_r, out_r = reduce_pca(train, in_dist, out_dist, num_comp)
elif reduce_type == 'tsne':
  train_r, in_r, out_r = reduce_tsne(train, in_dist, out_dist, num_comp)
elif reduce_type == 'umap':
  train_r, in_r, out_r = reduce_umap(train, in_dist, out_dist, num_comp)
  
# Save the embeddings
save_embeddings(train_out, train_files, train_r, True)
save_embeddings(ID_out, in_files, in_r, True)
save_embeddings(OOD_out, out_files, out_r, True)
