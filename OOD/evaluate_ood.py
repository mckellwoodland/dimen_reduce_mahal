"""
Evaluate the AUROC, AUPR, and FPR75 for given Mahalanobis distances.

The distances must be in three csv files: train_distances.csv, test_in_distances.csv, test_out_distances.csv.
These csv files must contain a column with the distances entitled 'Mahalanobis Distances'.
"""

# Imports
import argparse
import os
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('result_dir', type=str, help='Path to folder containing the CSVs with the Mahalanobis distances')
args = parser.parse_args()
result_fold = args.result_dir

# Functions
def auc(in_dist, out_dist):
    """
    Calculate the AUROC and AUPR for given Mahalanobis distances.
    Inputs:
      in_dist (pandas series): Mahalanobis distances for the in-distribution test data.
      out_dist (pandas series): Mahalanobis distances for the OOD test data.
    Returns:
        auroc (float): the AUROC
        aupr (float): the AUPR
    """
    # Create the ground truth. 0 ~ ID. 1 ~ OOD.
    truth = [0]*len(in_dist) + [1]*len(out_dist)
    # Normalize the data between 0 and 1.
    overall_max = max(max(in_dist), max(out_dist))
    # Combine the distances.
    pred = pd.concat([in_dist, out_dist]) / overall_max
    # Calculate the fprs and tprs.
    fpr, tpr, thresholds = metrics.roc_curve(truth, pred)
    # Calculate the AUROC.
    auroc = metrics.auc(fpr, tpr)
    # Calculate the AUPR.
    aupr = metrics.average_precision_score(truth, pred)
    return auroc, aupr

# Read in the Mahalanobis distances
train = pd.read_csv(os.path.join(result_fold, 'train_distances.csv')).drop(columns='Unnamed: 0')
in_test = pd.read_csv(os.path.join(result_fold, 'test_in_distances.csv')).drop(columns='Unnamed: 0')
out_test = pd.read_csv(os.path.join(result_fold, 'test_out_distances.csv')).drop(columns='Unnamed: 0')

# Calculate AUROC and AUPR
auroc, aupr = auc(in_test['Mahalanobis Distances'], out_test['Mahalanobis Distances'])

# Calculate FPR75
q = out_test['Mahalanobis Distances'].quantile(q=0.25)
print(sum(in_test['Mahalanobis Distances'] > q))
fpr75 = sum(in_test['Mahalanobis Distances'] > q) / len(in_test)

print(f"AUROC: {auroc}\nAUPR: {aupr}\nFPR75: {fpr75}")
