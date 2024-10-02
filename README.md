# Dimensionality Reduction and Nearest Neighbors for Improving Out-of-Distribution Detection in Medical Image Segmentation - Official Repository

<p align="center"><img src="https://github.com/mckellwoodland/dimen_reduce_mahal/blob/main/figures/graph_abstract.png" width="750" alt="Mahalanobis and k-th nearest neighbor distances pipelines with dimensionality-reduced features using either principal component analysis, t-distributed stochastic embeddings, uniform manifold approximation and projection, or average pooling. The encoder is a trained encoder from a U-Net architecture. k is the k-th nearest neighbor."</p>
 
**Dimensionality Reduction and Nearest Neighbors for Improving Out-of-Distribution Detection in Medical Image Segmentation**  
McKell Woodland, Nihil Patel, Austin Castelo, Mais Al Taie, Mohamed Eltaher, Joshua P. Yung, Tucker J. Netherton, Tiffany L. Calderone, Jessica I. Sanchez, Darrel W. Cleere, Ahmed Elsaiey, Nakul Gupta, David Victor, Laura Beretta, Ankit B. Patel, & Kristy K. Brock

Abstract: *Clinically deployed deep learning-based segmentation models are known to fail on data outside of their training distributions. While clinicians review the segmentations, these models tend to perform well in most instances, which could exacerbate automation bias. Therefore, detecting out-of-distribution images at inference is critical to warn the clinicians that the model likely failed. This work applied the Mahalanobis distance (MD) post hoc to the bottleneck features of four Swin UNETR and nnU-net models that segmented the liver on T1-weighted magnetic resonance imaging and computed tomography. By reducing the dimensions of the bottleneck features with either principal component analysis or uniform manifold approximation and projection, images the models failed on were detected with high performance and minimal computational load. In addition, this work explored a non-parametric alternative to the MD, a k<sup>th</sup> nearest neighbors distance (KNN). KNN drastically improved scalability and performance over MD when both were applied to raw and average-pooled bottleneck features.*

This work is published in [MELBA](https://doi.org/10.59275/j.melba.2024-g93a), with a preprint available on [arXiv](https://arxiv.org/abs/2408.02761).
This work was first published in the [proceedings of the 2023 MICCAI UNSURE workshop](https://link.springer.com/chapter/10.1007/978-3-031-44336-7_15), where it won the best spotlight paper ([preprint](https://arxiv.org/abs/2308.03723)).
It was extended to include validation of the dimensionality reduction techniques for three additional liver segmentation models (including extensions to computed tomography and the nnU-net architecture), a novel analysis of the k<sup>th</sup> nearest neighbor distance (KNN) as a replacement for Mahalanobis distance (MD), and greater context into how MD and KNN fit into the larger out-of-distribution detection field by comparing their performance to standard methods.

Our initial release of code associated with the manuscript is available/citable via [Zenodo](https://doi.org/10.5281/zenodo.13881989).

# Docker

The docker container for our OOD code can be built and run with the following code:
```
docker build -t md_ood .
```
```
docker run -it --rm -v $(pwd):/workspace md_ood
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

Create a `data` folder in the SMIT repository with subfolders `imagesTr`, `imagesTs`, `labelsTr`, and `labelsTs`. Move all the images from the public datasets into `imagesTr` and the ground truth segmentations into `labelsTr`. For the AMOS dataset, we only used training and validation images with indices 507-600, as these are the MRIs. The labels need to be converted to binary masks of the liver. For the Duke Liver MRI dataset, the images associated with anonymized patient IDs 2, 3, 4, 9 (image 4 only), 10 (images 3 and 5 only), 11 (image 4 only), 17, 18 (image 3 only), 20, 23, 31, 32, 33, 35, 38, 42, 46, 50, 61 (image 2205 only), 63, 74, 75, 78, 83, and 84 were discarded due to either missing liver segments or poor image quality to the point where it was hard to identify the liver.

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
  --base_dir BASE_DIR  Path to the directory that contains the 'amos22', 'imagesTr', and 'labelsTr' folders.

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

In the `data` folder of the SMIT repository, create folders `imagesTs` and `labelsTs`. Your testing images should go into `imagesTs` and masks into `labelsTs`. We tested our model on institutional datasets.

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

# OOD Detection

## Embeddings extraction

Once trained, save off embeddings for all train and test images as `pt` files. This can be done using the `get_encodings.py` file in the forked SMIT repository.

Train, in-distribution (ID) test, and out-of-distribution (OOD) test embeddings should be put in different folders. 
For our work, ID is distinguished from OOD using the performance of the segmentation model. 
I.e. >95% Dice similarity coefficient (DSC) is ID, whereas <95% is OOD.

## Mahalanobis distance

```
usage: OOD/calc_md.py [-h] -tr TRAIN_PTH -te TEST_PTH -p PER_SEG_PTH
                  [-cm CORRELATION_METRIC] [-cp CORRELATION_PTH] [-d DIM]
                  [-e EMBED_TYPE] [-fc FILE_COL] [-k KERNEL_SIZE]
                  [-nc NAME_COL] [-n NUM_COMP] [-o OUT_FILE] [-r REDUCE_TYPE]
                  [-sa SAVE] [-sl SLIDING_WINDOW] [-st STRIDE]
                  [-t THRES]

Required Arguments:
  -tr TRAIN_PTH, --train_pth TRAIN_PTH
                        Path to folder containing training embeddings.
  -te TEST_PTH, --test_pth TEST_PTH
                        Path to folder containing testing embeddings.
  -p PER_SEG_PTH, --per_seg_pth PER_SEG_PTH
                        Path to CSV files with segmentation performance
                        results.

Optional Arguments:
  -cm CORRELATION_METRIC, --correlation_metric CORRELATION_METRIC
                        Name of the column in the CSV file pointed to by
                        correlation_path that contains the variable to
                        calculate the coefficient on. Defaults to None
  -cp CORRELATION_PTH, --correlation_pth CORRELATION_PTH
                        A path to a CSV file to calculate the Pearson
                        correlation coefficient. Filenames must be under
                        the file_col parameter. Defaults to None.
  -d DIM, --dim DIM     2- or 3-dimensional average pooling reduction
                        [2D][3D]. If all average pooling parameters are not given,
                        a hyperparameter search will be performed. Defaults to
                        None.
  -e EMBED_TYPE, --embed_type EMBED_TYPE
                        Whether the embeddings were saved as Torch embeddings
                        in .pt files [torch] or as NumPy embeddings in .npy
                        files [numpy]. Defaults to torch.
  -fc FILE_COL, --file_col FILE_COL
                        Name of the column in the segmentation performance CSV
                        file that contains the filenames. Defaults to
                        filename.
  -k KERNEL_SIZE, --kernel_size KERNEL_SIZE
                        Kernel size for average pooling reduction. If all
                        average pooling parameters are not given, a hyperparameter
                        search will be performed. Defaults to None.
  -nc NAME_COL, --name_col NAME_COL
                        Name of the column in the segmentation performance CSV
                        file that contains the dice scores. Defaults to
                        DICE_Score.
  -n NUM_COMP, --num_comp NUM_COMP
                        Number of components to use with dimensionality-
                        reduction technique. If not given, a hyperparameter
                        search will be performed. Defaults to None.
  -o OUT_FILE, --out_file OUT_FILE
                        txt filename to write output to.
  -r REDUCE_TYPE, --reduce_type REDUCE_TYPE
                        The dimensionality reduction technique: no reduction
                        [none], principal component analysis [pca],
                        t-distributed stochastic neighbor embeddings [tsne],
                        uniform manifold and projection [umap], and average
                        pooling [avgpool]. If None is given, a hyperparameter
                        search over all techniques will be performed. Defaults
                        to None.
  -sa SAVE, --save SAVE
                        Path to save the distances to. If None is given, the
                        distances won't be saved. Defaults to None.
  -sl SLIDING_WINDOW, --sliding_window SLIDING_WINDOW
                        Whether to reduce the sliding window dimension with
                        average pooling [avg] or max pooling [max]. Defaults
                        to no reduction [None].
  -st STRIDE, --stride STRIDE
                        Stride for average pooling reduction. If all average
                        pooling parameters are not given, a hyperparameter search
                        will be performed. Defaults to None.
  -t THRES, --thres THRES
                        Dice similarity coefficient threshold to determine
                        which images are out-of-distribution. Defaults to None
                        (median value will be chosen). A value of None will
                        result in a median threshold.

```

## K<sup>th</sup> Nearest Neighbor

```
usage: OOD/calc_knn.py [-h] -tr TRAIN_PTH -te TEST_PTH [-o OUT_FILE] -p PER_SEG_PTH [-cm CORRELATION_METRIC] [-cp CORRELATION_PTH] [-d DIM] [-e EMBED_TYPE]
                   [-fc FILE_COL] [-k K] [-ks KERNEL_SIZE] [-nc NAME_COL] [-n NUM_COMP] [-r REDUCE_TYPE] [-sl SLIDING_WINDOW] [-st STRIDE] [-t THRES]

Required Arguments:
  -tr TRAIN_PTH, --train_pth TRAIN_PTH
                        Path to folder containing training embeddings.
  -te TEST_PTH, --test_pth TEST_PTH
                        Path to folder containing testing embeddings.
  -o OUT_FILE, --out_file OUT_FILE
                        txt filename to write output to.
  -p PER_SEG_PTH, --per_seg_pth PER_SEG_PTH
                        Path to CSV files with segmentation performance results.

Optional Arguments:
  -cm CORRELATION_METRIC, --correlation_metric CORRELATION_METRIC
                        Name of the column in the CSV file pointed to by correlation_path that contains the variable to calculate the coefficient on. Defaults to
                        None
  -cp CORRELATION_PTH, --correlation_pth CORRELATION_PTH
                        A path to a CSV file to calculate the Pearson correlation coefficient. Filenames must be under the file_col parameter. Defaults to
                        None.
  -d DIM, --dim DIM     2- or 3-dimensional average pooling reduction [2D][3D]. If all average pooling parameters are not given, a hyperparameter search will
                        be performed. Defaults to None.
  -e EMBED_TYPE, --embed_type EMBED_TYPE
                        Whether the embeddings were saved as Torch embeddings in .pt files [torch] or as NumPy embeddings in .npy files [numpy]. Defaults
                        to torch.
  -fc FILE_COL, --file_col FILE_COL
                        Name of the column in the segmentation performance CSV file that contains the filenames. Defaults to filename.
  -k K                  k in the k-th nearest neighbor distance. If no k is given, a hyperparameter search will be performed. Defaults to None.
  -ks KERNEL_SIZE, --kernel_size KERNEL_SIZE
                        Kernel size for average pooling reduction. If all average pooling parameters are not given, a hyperparameter search will be performed.
                        Defaults to None.
  -nc NAME_COL, --name_col NAME_COL
                        Name of the column in the segmentation performance CSV file that contains the dice scores. Defaults to DICE_Score.
  -n NUM_COMP, --num_comp NUM_COMP
                        Number of components to use with dimensionality-reduction technique. If not given, a hyperparameter search will be performed.
                        Defaults to None.
  -r REDUCE_TYPE, --reduce_type REDUCE_TYPE
                        The dimensionality reduction technique: no reduction [none], principal component analysis [pca], t-distributed stochastic neighbor
                        embeddings [tsne], uniform manifold and projection [umap], and average pooling [avgpool]. If None is given, a hyperparameter search
                        over all techniques will be performed. Defaults to None.
  -sl SLIDING_WINDOW, --sliding_window SLIDING_WINDOW
                        Whether to reduce the sliding window dimension with average pooling [avgpool] or max pooling [maxpool]. Defaults to no reduction
                        [None].
  -st STRIDE, --stride STRIDE
                        Stride for average pooling reduction. If all average pooling parameters are not given, a hyperparameter search will be performed.
                        Defaults to None.
  -t THRES, --thres THRES
                        Dice similarity coefficient threshold to determine which images are out-of-distribution. Defaults to 0.95. A value of None will
                        result in a median threshold.
```

# Maximum Softmax Probability, Temperature Scaling, and Energy Scoring

```
usage: OOD/calc_msp.py [-h] -te TEST_PTH [-o OUT_FILE] -p PER_SEG_PTH [-c CALIBRATION] [-cm CORRELATION_METRIC] [-cp CORRELATION_PTH] [-e EMBED_TYPE]
                   [-fc FILE_COL] [-nc NAME_COL] [-t THRES] [-tem TEMPERATURE]

Required Arguments:
  -te TEST_PTH, --test_pth TEST_PTH
                        Path to folder containing testing logits.
  -o OUT_FILE, --out_file OUT_FILE
                        txt filename to write output to.
  -p PER_SEG_PTH, --per_seg_pth PER_SEG_PTH
                        Path to CSV files with segmentation performance results.

Optional Arguments:
  -c CALIBRATION, --calibration CALIBRATION
                        Whether to use temperature scaling [ts] or energy [energy]. Defaults to None.
  -cm CORRELATION_METRIC, --correlation_metric CORRELATION_METRIC
                        Name of the column in the CSV file pointed to by correlation_path that contains the variable to calculate the coefficient on. Defaults to
                        None
  -cp CORRELATION_PTH, --correlation_pth CORRELATION_PTH
                        A path to a CSV file to calculate the Pearson correlation coefficient. Filenames must be under the file_col parameter. Defaults to
                        None.
  -e EMBED_TYPE, --embed_type EMBED_TYPE
                        Whether the embeddings were saved as NumPy (.npy) [numpy] or NifTI (.nii.gz) [nifti] embeddings. Defaults to numpy.
  -fc FILE_COL, --file_col FILE_COL
                        Name of the column in the segmentation performance CSV file that contains the filenames. Defaults to filename.
  -nc NAME_COL, --name_col NAME_COL
                        Name of the column in the segmentation performance CSV file that contains the dice scores. Defaults to DICE_Score.
  -t THRES, --thres THRES
                        Dice similarity coefficient threshold to determine which images are out-of-distribution. Defaults to 0.95. A value of None will
                        result in a median threshold.
  -tem TEMPERATURE, --temperature TEMPERATURE
                        Temperature for temperature scaling and energy scoring. Defaults to None. A value of None will result in a temperature search.
```

## Ensembling and Monte-Carlo Dropout

Save five outputs per test image with the filename suffix `_output{i}.nii.gz` for $i\in\{1,2,3,4,5\}$.

```
usage: OOD/calc_ensemble.py [-h] -te TEST_PTH [-o OUT_FILE] -p PER_SEG_PTH [-cm CORRELATION_METRIC] [-cp CORRELATION_PTH] [-fc FILE_COL] [-nc NAME_COL]
                        [-t THRES]

Required Arguments:
  -te TEST_PTH, --test_pth TEST_PTH
                        Path to folder containing testing outputs.
  -o OUT_FILE, --out_file OUT_FILE
                        txt filename to write output to.
  -p PER_SEG_PTH, --per_seg_pth PER_SEG_PTH
                        Path to CSV files with segmentation performance results.

Optional Arguments:
  -cm CORRELATION_METRIC, --correlation_metric CORRELATION_METRIC
                        Name of the column in the CSV file pointed to by correlation_path that contains the variable to calculate the coefficient on. Defaults to
                        None
  -cp CORRELATION_PTH, --correlation_pth CORRELATION_PTH
                        A path to a CSV file to calculate the Pearson correlation coefficient. Filenames must be under the file_col parameter. Defaults to
                        None.
  -fc FILE_COL, --file_col FILE_COL
                        Name of the column in the segmentation performance CSV file that contains the filenames. Defaults to filename.
  -nc NAME_COL, --name_col NAME_COL
                        Name of the column in the segmentation performance CSV file that contains the dice scores. Defaults to DICE_Score.
  -t THRES, --thres THRES
                        Dice similarity coefficient threshold to determine which images are out-of-distribution. Defaults to 0.95. A value of None will
                        result in a median threshold.
```

# Citation

If you have found our work useful, we would appreciate citations to our manuscripts and code.
```
@software{woodland_2024_13881989,
  author       = {Woodland, McKell and
                  Patel, Ankit B. and
                  Brock, Kristy K.},
  title        = {{Dimensionality Reduction and Nearest Neighbors for 
                   Improving Out-of-Distribution Detection in Medical
                   Image Segmentation - Official Repository}},
  month        = oct,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.13881989},
  url          = {https://doi.org/10.5281/zenodo.13881989}
}
```
```
@article{melba:2024:020:woodland,
    title = "Dimensionality Reduction and Nearest Neighbors for Improving Out-of-Distribution Detection in Medical Image Segmentation",
    author = "Woodland, McKell and Patel, Nihil and Castelo, Austin and Al Taie, Mais and Eltaher, Mohamed and Yung, Joshua P. and Netherton, Tucker J. and Calderone, Tiffany L. and Sanchez, Jessica I. and Cleere, Darrel W. and Elsaiey, Ahmed and Gupta, Nakul and Victor, David and Beretta, Laura and Patel, Ankit B. and Brock, Kristy K.",
    journal = "Machine Learning for Biomedical Imaging",
    volume = "2",
    issue = "UNSURE2023 special issue",
    year = "2024",
    pages = "2006--2052",
    issn = "2766-905X",
    doi = "https://doi.org/10.59275/j.melba.2024-g93a",
    url = "https://melba-journal.org/"
}
```
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

Research reported in this publication was supported in part by the Tumor Measurement Initiative through the MD Anderson Strategic Initiative Development Program (STRIDE), the Helen Black Image Guided Fund, the Image Guided Cancer Therapy Research Program at The University of Texas MD Anderson Cancer Center, a generous gift from the Apache Corporation, and the National Cancer Institute of the National Institutes of Health under award numbers R01CA221971, R01CA235564, R01CA195524, and P30CA016672. We'd like to thank Dr. Eugene J. Koay, Ph.D., for providing the liver MRI data from MD Anderson and Sarah Bronson - Scientific Editor at the Research Medical Library at MD Anderson - for editing sections of this article.
