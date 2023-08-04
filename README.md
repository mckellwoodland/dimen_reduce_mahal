# Dimensionality Reduction for Mahalanobis Distance-based OOD Detection - Official Repository

<p align='center'>
  <img src="https://github.mdanderson.org/Morfeus-Lab/swin_unetr_med_ood/blob/main/figures/figure3.png" width="300" height="300">
 </p>

**Dimensionality Reduction for Improving Out-of-Distribution Detection in Medical Image Segmentation**  
M. Woodland, N. Patel, M. Al Taie, J.P. Yung, T.J. Netherton, A.B. Patel, & K.K. Brock

Abstract: *Clinically deployed segmentation models are known to fail on data outside of their training distribution.
As these models perform well on most cases, it is imperative to detect out-of-distribution images at inference to protect against automation bias. 
This work applies the Mahalanobis distance post hoc to the bottleneck features of a Swin UNETR model that segments the liver on T1-weighted magnetic resonance imaging.
By reducing the dimensions of the bottleneck features with principal component analysis, images the model failed on were detected with high performance (92\% area under the receiver operating characteristic curve and 94\% area under the precision-recall curve) and minimal computational load.*

## Train the segmentation model
[Link to code for training the segmentation model](https://github.mdanderson.org/Morfeus-Lab/SMIT-Liver-Segmentation)

Save off embeddings for all train and test images as `pt` files using `get_encodings.py`. Train, in-distribution (ID) test, and out-of-distribution (OOD) test embeddings should be put in different folders. ID is distinguished from OOD using the performance of the segmentation model. I.e. >95% Dice similarity coefficient (DSC) is ID, whereas <95% is OOD.
 
## OOD detection set up

Build and run the Docker container.
```
docker build -t swin_unetr_ood .
```

## Reduce the dimensions of the embeddings.

Reduce the dimensionality of the embeddings using average pooling, PCA, t-SNE, or UMAP.
Encodings must be '.pt' files.

 ```
 usage: reduce_dim.py [-h] --train_in TRAIN_IN --train_out TRAIN_OUT --ID_in
                     ID_IN --ID_out ID_OUT --OOD_in OOD_IN --OOD_out OOD_OUT
                     --type TYPE [--num_comp NUM_COMP] [--kernel KERNEL]
                     [--stride STRIDE] [--dim DIM]

optional arguments:
  -h, --help            show this help message and exit
  --train_in TRAIN_IN   Path to folder containing the input training
                        embeddings
  --train_out TRAIN_OUT
                        Path to folder to contain the output training
                        embeddings
  --ID_in ID_IN         Path to folder containing the input in-distribution
                        test embeddings
  --ID_out ID_OUT       Path to folder to contain the output in-distribution
                        test embeddings
  --OOD_in OOD_IN       Path to folder containing the input out-of-
                        distribution test embeddings
  --OOD_out OOD_OUT     Path to folder to contain the output out-of-
                        distribution test embeddings
  --type TYPE           Type of dimensionality reduction to use: "avgpool",
                        "pca", "tsne", or "umap"
  --num_comp NUM_COMP   Number of components to use for PCA, t-SNE, or UMAP
  --kernel KERNEL       Kernel size for average pooling
  --stride STRIDE       Stride size for average pooling
  --dim DIM             Dimensionality of pooling for average pooling: "2D" or
                        "3D"
 ```

## Calculate the Mahalanobis distance.

Calculate the Mahalanobis distance for given embeddings.

```
usage: mahalanobis_distance.py [-h] --train_in TRAIN_IN --ID_in ID_IN --OOD_in
                               OOD_IN --result RESULT

optional arguments:
  -h, --help           show this help message and exit
  --train_in TRAIN_IN  Path to folder containing the training embeddings
  --ID_in ID_IN        Path to folder containing the in-distribution test
                       embeddings
  --OOD_in OOD_IN      Path to folder containing the out-of-distribution test
                       embeddings
  --result RESULT      Path to folder to put the resulting distances into
```

## Evaluate the OOD detection.

Evaluate the AUROC, AUPR, and FPR75 for given Mahalanobis distances.

The distances must be in three csv files: train_distances.csv, test_in_distances.csv, and test_out_distances.csv.
These csv files must contain a column with the distances entitled 'Mahalanobis Distances'.

```
usage: evaluate_ood.py [-h] result_dir

positional arguments:
  result_dir  Path to folder containing the CSVs with the Mahalanobis
              distances.

optional arguments:
  -h, --help  show this help message and exit
```
