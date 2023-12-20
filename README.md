# Dimensionality Reduction for Improved Mahalanobis Distance: A Strategy for Out-of-Distribution Detection in Medical Image Segmentation - Official Repository

<p align="center"><img src="https://github.com/mckellwoodland/dimen_reduce_mahal/blob/main/figures/figure2_github.png" width="512" alt="The top row contains ID images with the highest Dice Similarity Coefficients (DSCs). The bottom row contains OOD images with the lowest DSCs. The Mahalanobis distances are also shown."</p>
 
**Dimensionality Reduction for Improved Mahalanobis Distance: A Strategy for Out-of-Distribution Detection in Medical Image Segmentation**  
McKell Woodland, Nihil Patel, Mais Al Taie, Joshua P. Yung, Tucker J. Netherton, Ankit B. Patel, & Kristy K. Brock

Abstract: *Clinically deployed deep learning-based segmentation models are known to fail on data outside of their training distributions. While clinicians review the segmentations, these models do tend to perform well in most instances, which could exacerbate automation bias. Therefore, it is critical to detect out-of-distribution images at inference to warn the clinicians that the model likely failed. This work applies the Mahalanobis distance post hoc to the bottleneck features of a Swin UNETR model that segments the liver on T1-weighted magnetic resonance imaging. By reducing the dimensions of the bottleneck features with principal component analysis, images the model failed on were detected with high performance and minimal computational load. Specifically, the proposed technique achieved 92% area under the receiver operating characteristic curve and 94% area under the precision-recall curve and can run in seconds on a central processing unit.*

Original article was published in the proceedings of the 2023 MICCAI UNSURE workshop and is available through [Springer](https://link.springer.com/chapter/10.1007/978-3-031-44336-7_15). Preprint of the expansion is available on [arXiv](https://arxiv.org/abs/2308.03723).

## Segmentation Model

Train the segmentation model. Then, save off embeddings for all train and test images as `pt` files. Train, in-distribution (ID) test, and out-of-distribution (OOD) test embeddings should be put in different folders. ID is distinguished from OOD using the performance of the segmentation model. I.e. >95% Dice similarity coefficient (DSC) is ID, whereas <95% is OOD.
 
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

# Citation

If you have found our work useful, we would appreciate a citation.
```
@InProceedings{10.1007/978-3-031-44336-7_15,
     author="Woodland, McKell
             and Patel, Nihil
             and Al Taie, Mais
             and Yung, Joshua P.
             and Netherton, Tucker J.
             and Patel, Ankit B.
             and Brock, Kristy K.",
     editor="Sudre, Carole H.
             and Baumgartner, Christian F.
             and Dalca, Adrian
             and Mehta, Raghav
             and Qin, Chen
             and Wells, William M.",
     title="Dimensionality Reduction for Improving Out-of-Distribution Detection in Medical Image Segmentation",
     booktitle="Uncertainty for Safe Utilization of Machine Learning in Medical Imaging",
     year="2023",
     publisher="Springer Nature Switzerland",
     address="Cham",
     pages="147--156",
     isbn="978-3-031-44336-7"
}
```

# Acknowledgments

Research reported in this publication was supported in part by the Tumor Measurement Initiative through the MD Anderson Strategic Initiative Development Program (STRIDE), the Helen Black Image Guided Fund, the Image Guided Cancer Therapy Research Program at The University of Texas MD Anderson Cancer Center, a generous gift from the Apache Corporation, and the National Cancer Institute of the National Institutes of Health under award numbers R01CA221971, P30CA016672, and R01CA235564.
