This directory contains the original output logs for the 976 configurations of the KNN hyperparameter search.
The logs are labeled `knn_{model_name}_{dimensionaly_reduction_configuration}_{performance_threshold}.txt`.
Options for `model_name` include MRI UNETR `mri_unetr`, MRI+ UNETR `mriplus_unetr`, MRI+ nnU-net `mriplus_nnunet`, and CT nnU-net `ct_nnunet`.
Options for `dimensionality_reduction_configuration` include no reduction `none`, PCA(n) `pca{n}` with n components, t-SNE `tsne`, UMAP(n) `umap{n}` with n components, and average pooling `avgpoold{d}k{k}s{s}` for dimension d kernel size k and stride s.
Options for `performance_threshold` are 80% `80` and 95% `95`.

The log files are formatted as follows

```
K: {k}
-------
AUROC: {mean} ({std})
[{individual AUROCs}]
AUPRC: {mean} ({std})
[{individual AUPRCs}]
FPR90: {mean} ({std})
[{individual FPR90s}]
Time: {mean} ({std})
[{individual seconds}]
Pearson: {mean} ({std})
[{individual Pearson correlation coefficients}]
p-values: [{correlation coefficient p-values}]
Rejected: {can be ignored} ({can be ignored})
[{can be ignored}]
```
