"""
Calculates OOD detection results using the k-th nearest neighbor distance on previously extracted features.

PyTorch encodings must be saved as .pt files.
NumPy encodings must be saved as .npy files.
"""

# Imports
import argparse
import monai
import time
import umap
import utils

import numpy as np
import pandas as pd

from scipy import linalg, stats
from scipy.spatial import distance
from sklearn import decomposition, manifold, preprocessing
from torch import nn

# Arguments
parser = argparse.ArgumentParser()
parser._action_groups.pop()

required = parser.add_argument_group('Required Arguments')
required.add_argument('-tr', '--train_pth', type=str, required=True, help='Path to folder containing training embeddings.')
required.add_argument('-te', '--test_pth', type=str, required=True, help='Path to folder containing testing embeddings.')
required.add_argument('-o', '--out_file', type=str, default=None, help='txt filename to write output to.')
required.add_argument('-p', '--per_seg_pth', type=str, required=True, help='Path to CSV files with segmentation performance results.')

optional = parser.add_argument_group('Optional Arguments')
optional.add_argument('-cm', '--correlation_metric', type=str, default=None, help='Name of column in csv file pointed to by correlation_path that contains the variable to calculate the coefficient on.\
                                                                                   Defaults to None')
optional.add_argument('-cp', '--correlation_pth', type=str, default=None, help='A path to a csv file to calculate the Pearson correlation coefficient with. \
                                                                                Filenames must be under file_col parameter.\
                                                                                Defaults to None.')
optional.add_argument('-d', '--dim', type=str, default=None, help='2- or 3-dimensional average pooling reduction [2D][3D].\
                                                                   If all average pooling params are not given, a hyperparameter search will be performed.\
                                                                   Defaults to None.')
optional.add_argument('-e', '--embed_type', type=str, default='torch', help='Whether the embeddings were saved as Torch embeddings in .pt files [torch] or as NumPy embeddings in .npy files [numpy]. \
                                                                             Defaults to torch.')
optional.add_argument('-fc', '--file_col', type=str, default='filename', help='Name of the column in the segmentation performance CSV file that contains the filenames.\
                                                                              Defaults to filename.')
optional.add_argument('-k', type=int, default=None, help='k in the k-th nearest neighbor distance.\
                                                          If no k is given, a hyperparameter search will be performed.\
                                                         Defaults to None.')
optional.add_argument('-ks', '--kernel_size', type=int, default=None, help='Kernel size for average pooling reduction.\
                                                                   If all average pooling params are not given, a hyperparameter search will be performed.\
                                                                        Defaults to None.')
optional.add_argument('-nc', '--name_col', type=str, default='DICE_Score', help='Name of the column in the segmentation performance CSV file that contains the dice scores.\
                                                                                Defaults to DICE_Score.')
optional.add_argument('-n', '--num_comp', type=int, default=None, help='Number of components to use with dimensionality-reduction technique.\
                                                                        If not given, a hyperparameter search will be performed.\
                                                                        Defaults to None.')
optional.add_argument('-r', '--reduce_type', type=str, default=None, help='The dimensionality reduction technique: no reduction [none], principal component analysis [pca], t-distributed stochastic neighbor embeddings [tsne], uniform manifold and projection [umap], and average pooling [avgpool]. \
                                                                            If None is given, a hyperparameter search over all techniques will be performed.\
                                                                           Defaults to None.')
optional.add_argument('-sl', '--sliding_window', type=str, default=None, help='Whether to reduce the sliding window dimension with average pooling [avgpool] or max pooling [maxpool]. \
                                                                              Defaults to no reduction [None].')
optional.add_argument('-st', '--stride', type=int, default=None, help='Stride for average pooling reduction.\
                                                                   If all average pooling params are not given, a hyperparameter search will be performed.\
                                                                    Defaults to None.')
optional.add_argument('-t', '--thres', type=float, default=0.95, help='Dice similarity coefficient threshold to determine which images are out-of-distribution. \
                                                                       Defaults to 0.95. \
                                                                       A value of None will result in a median threshold.')
optional.add_argument('-u', '--unaccept_pth', type=str, default=None, help='Path to folder containing embeddings of the clinically unnaceptable images.\
                                                                            Defaults to not being evaluated.')

args = parser.parse_args()

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
def kth_distance(test, train, k, test_f):
    """
    Calculate the k-th nearest neighbor distance.

    Input:
        test (list): Test embeddings.
        train (list): Train embeddings.
        k (int): Hyperparameter k.
        test_f (list): Test files.

    Return:
        Pandas dataframe with distances.
    """
    dists = distance.cdist(test,train)
    return pd.DataFrame({'dist':[sorted(row)[k] for row in dists],
                         f'{args.file_col}':[f.split('.')[0] for f in test_f]})

def eval(ID, ID_f, OOD, OOD_f, train, k_val=None, spcl=None, spcl_f=None, reduce_type=None, num_comp=None, dim=None, stride=None, kernel_size=None):
    """
    Evaluate the k-th nearest neighbor distance for a specified k and dimensionality-reduction setting.

    Inputs:
        ID (list): ID embeddings.
        ID_f (list): ID filenames.
        OOD (list): OOD embeddings.
        OOD_f (list): OOD filenames.
        train (list): Training embeddings.
        k_val (list): Hyperparameter k.
        spcl (list): Unacceptable embeddings.
        spcl_f (list): Unacceptable filenames.
        reduce_tye (str): Dimensionality reduction to be performed.
        num_comp (int): Number of components for PCA or UMAP.
        dim (str): 2D or 3D for average pooling.
        stride (int): Average pooling stride.
        kernel_size (int): Average pooling kernel size. 
    """
    if k_val:
        ks = [k_val]
    else:
        ks = [2**n for n in range(1,9)]
        
    for k in ks:
        AUROCs, AUPRCs, FPR90s, SECONDs, STATs, PVALs, REJECTs = [], [], [], [], [], [], []
        DSCs, HDs, SDs, numOODs = [], [], [], []
        if args.correlation_pth:
            df_c = pd.read_csv(args.correlation_pth)
            CORRs, CORR_ps = [], []
        for seed in range(5):
            np.random.seed(seed)
            
            start = time.time()
            if reduce_type == 'pca':
                train_reduce, dataframes = reduce_pca(train, [ID, OOD], num_comp)
                ID_reduce, OOD_reduce = dataframes
            elif reduce_type =='tsne':
                train_reduce, dataframes = reduce_tsne(train, [ID, OOD], 2)
                ID_reduce, OOD_reduce = dataframes[:len(ID_f)], dataframes[len(ID_f):]
            elif reduce_type == 'umap':
                train_reduce, dataframes = reduce_umap(train, [ID, OOD], num_comp)
                ID_reduce, OOD_reduce = dataframes[:len(ID_f)], dataframes[len(ID_f):]
            elif reduce_type == 'avgpool':
                train_reduce = reduce_avgpool(train, kernel_size, stride, dim)
                ID_reduce = reduce_avgpool(ID, kernel_size, stride, dim)
                OOD_reduce = reduce_avgpool(OOD, kernel_size, stride, dim)
            else:
                train_reduce, ID_reduce, OOD_reduce = train, ID, OOD
            ID_dists = kth_distance(ID_reduce, train_reduce, k, ID_f)
            OOD_dists = kth_distance(OOD_reduce, train_reduce, k, OOD_f)
            thres90 = OOD_dists['dist'].quantile(q=0.10)
            seconds = time.time() - start
            auroc, auprc, fpr90 = utils.evaluate(ID_dists['dist'], OOD_dists['dist'])
            df_test = pd.concat([ID_dists, OOD_dists])
            df_test = df_test.merge(df, how='inner', on=args.file_col)
            statistic, pvalue = stats.pearsonr(df_test.dist, df_test[args.name_col])
            AUROCs.append(auroc)
            AUPRCs.append(auprc)
            FPR90s.append(fpr90)
            SECONDs.append(seconds)
            STATs.append(statistic)
            PVALs.append(pvalue)
            DSCs.append(df_test[df_test['dist'] <= thres90][args.name_col].mean() - dsc_mean)
            HDs.append(df_test[df_test['dist'] <= thres90]['HD'].mean() - hd_mean)
            SDs.append(df_test[df_test['dist'] <= thres90]['SD'].mean() - sd_mean)
            numOODs.append(len(df_test[df_test['dist'] > thres90]))
            if args.correlation_pth:
                if args.correlation_metric in df_test.columns:
                    df_test = df_test.drop(args.correlation_metric,axis=1)
                df_comb = df_test.merge(df_c,how='inner',on=args.file_col)
                stat, p = stats.pearsonr(df_comb.dist, df_comb[args.correlation_metric])
                CORRs.append(stat)
                CORR_ps.append(p)
        with open(args.out_file, 'a') as f:
            f.write(f"K: {k}\n--------\n")
            f.write(f"Reduction: {reduce_type} {num_comp} {kernel_size} {stride} {dim}\n")
            f.write(f"AUROC: {round(np.mean(AUROCs),2)} ({round(np.std(AUROCs),2)})\n{AUROCs}\n")
            f.write(f"AUPRC: {round(np.mean(AUPRCs),2)} ({round(np.std(AUPRCs),2)})\n{AUPRCs}\n")
            f.write(f"FPR90: {round(np.mean(FPR90s),2)} ({round(np.std(FPR90s),2)})\n{FPR90s}\n")
            f.write(f"Time: {round(np.mean(SECONDs),2)} ({round(np.std(SECONDs),2)})\n{SECONDs}\n")
            f.write(f"Pearson: {round(np.mean(STATs),2)} ({round(np.std(STATs),2)})\n{STATs}\n")
            f.write(f"p-values: {PVALs}\n")
            if args.correlation_pth:
                f.write(f"Pearson with {args.correlation_metric}: {round(np.mean(CORRs),2)} ({round(np.std(CORRs),2)})\n{CORRs}\n{CORR_ps}\n")
            f.write(f"DSC: {round(np.mean(DSCs),2)} ({round(np.std(DSCs),2)})\n")
            f.write(f"HD: {round(np.mean(HDs),2)} ({round(np.std(HDs),2)})\n")
            f.write(f"SD: {round(np.mean(SDs),2)} ({round(np.std(SDs),2)})\n")
            f.write(f"#OOD: {round(np.mean(numOODs),2)} ({round(np.std(numOODs),2)})\n")

def reduce_avgpool(dataframe, kernel_size, stride, dim):
    """
   Perform average pooling on embedding to reduce dimensionality.
    Inputs:
        dataframe (array): Embeddings to be pooled.
        kernel_size (int): Size of the pooling kernel
        stride (int): Stride of the pooling kernel
        dim (str): either '2D' or '3D' for dimensionality of the embedding
    Returns:
        train_avgpool (array): training embeddings with dimensionality reduced by PCA
        ID_avgpool (array): in-distribution test embeddings with dimensionality reduced by PCA
        OOD_avgpool (array): OOD test embeddings with dimensionality reduced by PCA
    """
    model = AvgPool(kernel_size, stride, dim)
    return np.array([model(x).numpy().flatten() for x in dataframe])

def reduce_pca(train, dataframes, num_comp):
    """
    Reduce the dimensionality of the embeddings with Principal Component Analysis (PCA).
    Inputs:
        train (array): Training embeddings.
        dataframes (list): List of embedding datasets (i.e. in-distribution, out-of-distribution, etc.)
        num_comp (int): The number of components for PCA.
    Outputs:
        train (array): Reduced training embeddings.
        dataframes (list): List of reduced embeddings datasets.
    """
    scale = preprocessing.StandardScaler()
    train = scale.fit_transform(train)
    dataframes = [scale.transform(frame) for frame in dataframes]

    pca = decomposition.PCA(n_components=num_comp)
    train = pca.fit_transform(train)
    dataframes = [pca.transform(frame) for frame in dataframes]
    return train, dataframes

def reduce_tsne(train, dataframes, num_comp):
    """
    Reduce the dimensionality of the embeddings with t-Distributed Stochastic Neighbor Embedding (t-SNE).
    Inputs:
        train (array): Training embeddings.
        dataframes (list): List of embedding datasets (i.e. in-distribution, out-of-distribution, etc.)
        num_comp (int): The number of components.
    Outputs:
        Reduced training embeddings.
        Reduced embeddings datasets.
    """
    combine = np.vstack([train]+dataframes)
    model = manifold.TSNE(num_comp)
    combine_t = model.fit_transform(combine)
    return combine_t[:len(train)], combine_t[len(train):]

def reduce_umap(train, dataframes, num_comp):
    """
    Reduce the dimensionality of the embeddings with Uniform Manifold Approximation and Projection (UMAP).
    Inputs:
        train (array): Training embeddings.
        dataframes (list): List of embedding datasets (i.e. in-distribution, out-of-distribution, etc.)
        num_comp (int): The number of oomponents.
        seed (int): Random seed for UMAP.
                    Defaults to None.
    Outputs:
        Reduced training embeddings.
        Reduced embeddings datasets.
    """
    combine = np.vstack([train]+dataframes)
    model = umap.UMAP(n_components=num_comp)
    combine_t = model.fit_transform(combine)
    train_start, train_end = 0, len(train)
    return combine_t[:len(train)], combine_t[len(train):]

def split_test(test_f, df, thres):
    """
    Split test data into ID and OOD.

    Inputs:
        test_f (list): Test filenames.
        df (pd): Pandas DataFrame with DSCs.
        thres (int): Threshold to split on.

    Returns:
        ID (list): ID embeddings.
        ID_f (list): ID filenames.
        OOD (list): OOD embeddings.
        OOD_f (list): OOD filenames.
    """
    ID, ID_f = [], []
    OOD, OOD_f = [], []
    for i, f in enumerate(test_f):
        basename = f.split('.')[0]
        if df[df[args.file_col] == basename][args.name_col].iloc[0] > thres:
            ID.append(test[i])
            ID_f.append(f)
        else:
            OOD.append(test[i])
            OOD_f.append(f)
    print(f'Number of ID images: {len(ID_f)}')
    print(f'Number of OOD images: {len(OOD_f)}')
    return ID, ID_f, OOD, OOD_f

if __name__=="__main__":
    # Calculate threshold.
    df = pd.read_csv(args.per_seg_pth)
    dsc_mean = df[args.name_col].mean()
    hd_mean = df['HD'].mean()
    sd_mean = df['SD'].mean()
    if not args.thres:
        thres = df[args.name_col].median()
    else:
        thres = args.thres
        
    if not args.reduce_type == 'avgpool':
        # Read in embeddings.
        train, train_f = utils.read_embeddings(args.train_pth,
                                         embed_type = args.embed_type,
                                         is_numpy = True,
                                         sliding_window = args.sliding_window)
        test, test_f = utils.read_embeddings(args.test_pth,
                                       embed_type = args.embed_type,
                                       is_numpy = True,
                                       sliding_window = args.sliding_window)

        # Split into ID/OOD.
        ID, ID_f, OOD, OOD_f = split_test(test_f, df, thres)

        # Calculate distances.
        if args.reduce_type == "none" or not args.reduce_type:
            eval(ID, ID_f, OOD, OOD_f, train, args.k)
        if args.reduce_type == "pca" or not args.reduce_type:
            if args.num_comp:
                eval(ID, ID_f, OOD, OOD_f, train, args.k, reduce_type="pca", num_comp=args.num_comp)
            else:
                for n in [2**n for n in range(1,9)]:
                    eval(ID, ID_f, OOD, OOD_f, train, args.k, reduce_type="pca", num_comp=n)
        if args.reduce_type == "tsne" or not args.reduce_type:
            eval(ID, ID_f, OOD, OOD_f, train, args.k, reduce_type="tsne")
        if args.reduce_type == "umap" or not args.reduce_type:
            if args.num_comp:
                eval(ID, ID_f, OOD, OOD_f, train, args.k, reduce_type="umap", num_comp=args.num_comp)
            else:
                for n in [2**n for n in range(1,9)]:
                    eval(ID, ID_f, OOD, OOD_f, train, args.k, reduce_type="umap", num_comp=n)
    if args.reduce_type == 'avgpool' or not args.reduce_type:
        # Read in embeddings.
        train, train_f = utils.read_embeddings(args.train_pth,
                                         embed_type = args.embed_type,
                                         is_numpy = False,
                                         sliding_window = args.sliding_window)
        test, test_f = utils.read_embeddings(args.test_pth,
                                       embed_type = args.embed_type,
                                       is_numpy = False,
                                       sliding_window = args.sliding_window)
        # Split into ID/OOD.
        ID, ID_f, OOD, OOD_f = split_test(test_f, df, thres)

        # Calculate distances.
        if args.dim and args.kernel_size and args.stride:
            eval(ID, ID_f, OOD, OOD_f, train, args.k, reduce_type="avgpool", dim=args.dim, kernel_size=args.kernel_size, stride=args.stride)
        else:
            for dim in ['2D','3D']:
                for kernel_size, stride in [(2,1),(2,2),(3,1),(3,2),(4,1)]:
                    eval(ID, ID_f, OOD, OOD_f, train, args.k, reduce_type="avgpool", dim=dim, kernel_size=kernel_size, stride=stride)

