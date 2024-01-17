# Dimensionality Reduction for Improving Out-of-Distribution Detection in Medical Image Segmentation - Official Repository

<p align="center"><img src="https://github.com/mckellwoodland/dimen_reduce_mahal/blob/main/figures/figure3.png" width="400" alt="The top row contains ID images with the highest Dice Similarity Coefficients (DSCs). The bottom row contains OOD images with the lowest DSCs. The Mahalanobis distances are also shown."</p>
 
**Dimensionality Reduction for Improving Out-of-Distribution Detection in Medical Image Segmentation**  
McKell Woodland, Nihil Patel, Mais Al Taie, Joshua P. Yung, Tucker J. Netherton, Ankit B. Patel, & Kristy K. Brock

Abstract: *Clinically deployed deep learning-based segmentation models are known to fail on data outside of their training distributions. While clinicians review the segmentations, these models do tend to perform well in most instances, which could exacerbate automation bias. Therefore, it is critical to detect out-of-distribution images at inference to warn the clinicians that the model likely failed. This work applies the Mahalanobis distance post hoc to the bottleneck features of a Swin UNETR model that segments the liver on T1-weighted magnetic resonance imaging. By reducing the dimensions of the bottleneck features with principal component analysis, images the model failed on were detected with high performance and minimal computational load. Specifically, the proposed technique achieved 92% area under the receiver operating characteristic curve and 94% area under the precision-recall curve and can run in seconds on a central processing unit.*

The article was published in the proceedings of the 2023 MICCAI UNSURE workshop and is available through [Springer](https://link.springer.com/chapter/10.1007/978-3-031-44336-7_15). The preprint is available on [arXiv](https://arxiv.org/abs/2308.03723).

# Docker

The docker container for our OOD code can be built and run with the following code:
```
docker build -t swin_unetr_ood .
```
```
docker run -it --rm -v $(pwd):/workspace swin_unetr_ood
```

The docker container to train the SMIT model and extract the features can be built and run with the following code:
```
docker build -t swin_unetr_smit SMIT/.
```
```
docker run -it --rm --gpus all -v $(pwd)/SMIT:/workspace swin_unetr_smit
```
To use this code, you must be on the `dimen_reduce_mahal` branch of the SMIT repository.


# Data
## Training 
Download the [AMOS](https://zenodo.org/records/7155725)<sup>1</sup>, [Duke Liver](https://zenodo.org/records/7774566)<sup>2</sup>, [CHAOS](https://zenodo.org/records/3431873)<sup>3</sup> datasets from Zenodo. You will need to request access to the Duke Liver dataset.

Create a `data` folder in the SMIT repository with subfolders `imagesTr`, `imagesTs`, `labelsTr`, and `labelsTs`. Move all the images from the public datasets into `imagesTr` and the ground truth segmentations into `labelsTr`. For the AMOS dataset, we only used training and validation images with indices 507-600 as these are the MRIs. The labels need to be converted to binary masks of the liver. For the Duke Liver MRI dataset, the images associated with anonymized patient IDs 2, 3, 4, 9 (image 4 only), 10 (images 3 and 5 only), 11 (image 4 only), 17, 18 (image 3 only), 20, 23, 31, 32, 33, 35, 38, 42, 46, 50, 61 (image 2205 only), 63, 74, 75, 78, 83, and 84 were discarded due to either missing liver segments or poor image quality to the point where it was hard to identify the liver.

If you would like to use our preprocessing code, unzip the public datasets into the `data` folder. When unzipped, you should see the `Train_Sets` (CHAOS), `amos22` (AMOS), and `7774566` (Duke Liver) folders in the `data` folder. You can then convert them with our docker container and the dataset preprocessing files in the `utils` folder: `preprocess_CHAOS.py`, `preprocess_AMOS.py`, and `preprocess_Duke.py`.

```
usage: preprocess_CHAOS.py [-h] --base_dir BASE_DIR [--min_val MIN_VAL] [--max_val MAX_VAL]

Required Arguments:
  --base_dir BASE_DIR  Path to directory that contains the 'Train_Sets', 'imagesTr', and 'labelsTr' folders.

Optional Arguments:
  --min_val MIN_VAL    Minimum pixel value that belongs to the contour to be extracted. Defaults to 55 (liver).
  --max_val MAX_VAL    Maximum pixel value that belongs to the contour to be extracted. Defaults to 70 (liver).
```

```
usage: preprocess_AMOS.py [-h] --base_dir BASE_DIR [--val VAL]

Required Arguments:
  --base_dir BASE_DIR  Path to directory that contains the 'amos22', 'imagesTr', and 'labelsTr' folders.

Optional Arguments:
  --val VAL            Voxel value that belongs to the contour to be extracted. Defaults to 6 (liver).
```

```
usage: preprocess_Duke.py [-h] --base_dir BASE_DIR [--discard DISCARD]

Required Arguments:
  --base_dir BASE_DIR  Path to directory that contains the '7774566', 'imagesTr', and 'labelsTr' folders.

Optional Arguments:
  --discard DISCARD    Whether to discard the images with poor quality. Defaults to True.
```

You'll need the `train.json` file in the `dataset` folder of the forked SMIT repository (branch `dimen_reduce_mahal`).
If you are not using the same training images and preprocessing code, you'll need to create your JSON file following the below pattern containing paths to your images and labels. Name the file `train.json` and put it in the `dataset` folder.

```
{
  "training": [
    {
      "image": "IMG_PATH1",
      "label": "LABEL_PATH1"
    },
    {
      "image": "IMG_PATH2",
      "label": "LABEL_PATH2"
    },
  ]
}
```

## Testing

In the `data` folder of the SMIT repository, create folders `imagesTs` and `labelsTs`. Your testing images should go into `imagesTs` and masks into `labelsTs`. We tested our model on institutional datasets. These datasets may be made available upon request, in compliance with institutional IRB requirements.

You'll need the `test.json` file in the `dataset` folder of the SMIT repository. Create the JSON file following the given pattern:

```
{
  "training": [],
  "validation": [
    {
      "image": "IMG_PATH1",
      "label": "LABEL_PATH1"
    },
    {
      "image": "IMG_PATH2",
      "label": "LABEL_PATH2"
    }
  ]
}
```

# Segmentation Model
## Training

Train the segmentation model using the `fine_tuning_swin_3d.py` file of the official SMIT repository. A fork of the SMIT repository is included as a submodule. Our changes to the repository can be found in the `dimen_reduce_mahal` branch. Changes include a Dockerfile and updating dependencies so the code can run with the Docker container.

```
usage: fine_tuning_swin_3D.py [-h] [--checkpoint CHECKPOINT] [--logdir LOGDIR]
                              [--pretrained_dir PRETRAINED_DIR]
                              [--data_dir DATA_DIR] [--json_list JSON_LIST]
                              [--pretrained_model_name PRETRAINED_MODEL_NAME]
                              [--save_checkpoint] [--max_epochs MAX_EPOCHS]
                              [--batch_size BATCH_SIZE]
                              [--sw_batch_size SW_BATCH_SIZE]
                              [--optim_lr OPTIM_LR] [--optim_name OPTIM_NAME]
                              [--reg_weight REG_WEIGHT] [--momentum MOMENTUM]
                              [--noamp] [--val_every VAL_EVERY]
                              [--distributed] [--world_size WORLD_SIZE]
                              [--rank RANK] [--dist-url DIST_URL]
                              [--dist-backend DIST_BACKEND]
                              [--workers WORKERS] [--model_name MODEL_NAME]
                              [--pos_embed POS_EMBED] [--norm_name NORM_NAME]
                              [--num_heads NUM_HEADS] [--mlp_dim MLP_DIM]
                              [--hidden_size HIDDEN_SIZE]
                              [--feature_size FEATURE_SIZE]
                              [--in_channels IN_CHANNELS]
                              [--out_channels OUT_CHANNELS] [--res_block]
                              [--conv_block] [--use_normal_dataset]
                              [--a_min A_MIN] [--a_max A_MAX] [--b_min B_MIN]
                              [--b_max B_MAX] [--space_x SPACE_X]
                              [--space_y SPACE_Y] [--space_z SPACE_Z]
                              [--roi_x ROI_X] [--roi_y ROI_Y] [--roi_z ROI_Z]
                              [--dropout_rate DROPOUT_RATE]
                              [--RandFlipd_prob RANDFLIPD_PROB]
                              [--RandRotate90d_prob RANDROTATE90D_PROB]
                              [--RandScaleIntensityd_prob RANDSCALEINTENSITYD_PROB]
                              [--RandShiftIntensityd_prob RANDSHIFTINTENSITYD_PROB]
                              [--infer_overlap INFER_OVERLAP]
                              [--lrschedule LRSCHEDULE]
                              [--warmup_epochs WARMUP_EPOCHS] [--resume_ckpt]
                              [--resume_jit] [--smooth_dr SMOOTH_DR]
                              [--smooth_nr SMOOTH_NR]
```

We trained the model with the following command:
```
python fine_tuning_swin_3D.py \
     --pretrained_dir Pre_trained/ \
     --data_dir data/ \
     --json_list dataset/train.json \
     --max_epochs 1000
```

## Testing

# OOD Detection

## Embeddings extraction

Once trained, save off embeddings for all train and test images as `pt` files. This can be done using the `get_encodings.py` file in the forked SMIT repository.

Train, in-distribution (ID) test, and out-of-distribution (OOD) test embeddings should be put in different folders. ID is distinguished from OOD using the performance of the segmentation model. I.e. >95% Dice similarity coefficient (DSC) is ID, whereas <95% is OOD.

## Embedding Dimensionality Reduction

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

The distances must be in three CSV files: train_distances.csv, test_in_distances.csv, and test_out_distances.csv.
These CSV files must contain a column with the distances entitled 'Mahalanobis Distances'.

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

# References
1. JI YUANFENG. (2022). Amos: A large-scale abdominal multi-organ benchmark for versatile medical image segmentation [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7155725
2. Macdonald, J. A., Zhu, Z., Konkel, B., Mazurowski, M., Wiggins, W., & Bashir, M. (2020). Duke Liver Dataset (MRI) v2 (2.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7774566
3. Ali Emre Kavur, M. Alper Selver, Oğuz Dicle, Mustafa Barış, & N. Sinem Gezer. (2019). CHAOS - Combined (CT-MR) Healthy Abdominal Organ Segmentation Challenge Data (v1.03) [Data set]. The IEEE International Symposium on Biomedical Imaging (ISBI), Venice, Italy. Zenodo. https://doi.org/10.5281/zenodo.3431873

# Acknowledgments

Research reported in this publication was supported in part by the Tumor Measurement Initiative through the MD Anderson Strategic Initiative Development Program (STRIDE), the Helen Black Image Guided Fund, the Image Guided Cancer Therapy Research Program at The University of Texas MD Anderson Cancer Center, a generous gift from the Apache Corporation, and the National Cancer Institute of the National Institutes of Health under award numbers R01CA221971, P30CA016672, and R01CA235564.
