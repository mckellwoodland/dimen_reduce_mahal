"""
Calculate the Mahalanobis distance for given embeddings.
"""

# Imports
import argparse
import monai
import os
import scipy
import time
import torch
import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, help='Path to folder containing the training embeddings')
parser.add_argument('--ID_dir', type=str, help='Path to folder containing the in-distribution test embeddings')
parser.add_argument('--OOD_dir', type=str, help='Path to folder containing the out-of-distribution test embeddings')
parser.add_argument('--result_dir', type=str, help='Path to folder to put the resulting distances into')
parser.add_argument('--cov_est', type=str, default='MLE', help='How to estimate the covariance matrix. \
                                                                      Options: MLE or MCD. \
                                                                      Default: MLE.')

args = parser.parse_args()
train_fold = args.train_dir
test_in_fold = args.ID_dir
test_out_fold = args.OOD_dir
result_fold = args.result_dir
if not os.path.exists(result_fold):
    os.mkdir(result_fold)

# Functions
def read_embeddings(folder):
    """
    Read in bottleneck features.
    Inputs:
        path (str): path to the folder that has the encodings.
    Returns:
        (nparray): all encodings
    """
    embeds = []
    for pt_file in os.listdir(folder):
        pt_path = os.path.join(folder, pt_file)
        encoding = torch.load(pt_path)
        np_encoding = encoding.numpy()
        embeds.append(np_encoding.flatten())
    return np.stack(embeds, axis=0)

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

# Read in train embeddings.
train_embed = read_embeddings(train_fold)

# Calculate the mean of the distribution
train_mean = np.mean(train_embed, axis=0, keepdims=True).squeeze()

print("Calculating matrices")
start = time.time()

if args.cov_est == 'MLE':
    train_cov = np.cov(train_embed.T)
elif args.cov_est == 'MCD':
    train_cov = MinCovDet().fit(train_embed).covariance_

print(f"Time to calculate covariance matrix for embedding: {int(time.time() - start)} seconds")

start = time.time()
train_inv_cov = scipy.linalg.inv(train_cov)
print(train_inv_cov)
print(f"Time to calculate inverse covariance matrix for embedding: {time.time() - start} seconds")
    
# Read in test embeddings.
test_in = read_embeddings(test_in_fold)
test_out = read_embeddings(test_out_fold)

# Calculate the Mahalanobis distance.
print("Calculating distances")
start = time.time()
train_mahal = [mahal_dist(x, train_mean, train_inv_cov) for x in train_embed]
in_mahal = [mahal_dist(x, train_mean, train_inv_cov) for x in test_in]
out_mahal = [mahal_dist(x, train_mean, train_inv_cov) for x in test_out]
print("Time to calculate distances", time.time()-start)

# Save distances.
df_train = pd.DataFrame({'Filename': os.listdir(train_fold), 'Mahalanobis Distances': train_mahal})
df_in = pd.DataFrame({'Filename': os.listdir(test_in_fold), 'Mahalanobis Distances': in_mahal})
df_out = pd.DataFrame({'Filename': os.listdir(test_out_fold), 'Mahalanobis Distances': out_mahal})

df_train.to_csv(os.path.join(result_fold, 'train_distances.csv'))
df_in.to_csv(os.path.join(result_fold, 'test_in_distances.csv'))
df_out.to_csv(os.path.join(result_fold, 'test_out_distances.csv'))

