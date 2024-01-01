"""
Prepares the CHAOS dataset for training.
"""

# Imports
import argparse
import dicom2nifti
import os
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

# Arguments
parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('Required Arguments')
required.add_argument('--base_dir', required=True, type=str, help="Path to directory that contains the 'Train_Sets', 'imagesTr', and 'labelsTr' folders.")
args = parser.parse_args()
base_dir = args.base_dir
if not os.path.exists(os.path.join(base_dir, 'Train_Sets')):
  raise FileNotFoundError("The CHAOS dataset is not in the given base directory")
if not os.path.exists(os.path.join(base_dir, 'imagesTr')):
  os.mkdir(os.path.join(base_dir, 'imagesTr'))
if not os.path.exists(os.path.join(base_dir, 'labelsTr')):
  os.mkdir(os.path.join(base_dir, 'labelsTr'))

# Functions
def png_to_nifti(folder, contour=True, val=63):
  """
  Converts a folder of PNG images to one NIfTI image.
  Optionally extracts one contour of many.
  
  Inputs:
    folder (str): Path to folder that contains the PNG images.
    contour (bool): Whether to extract one contour.
    val (int): Value of contour to extract.
               Defaults to 63.

  Output:
    img_nii (NIfTI): NIfTI image
  """
  imgs = []
  for img_name in sorted(os.listdir(folder)):
    # Read in PNG image.
    img_pil = Image.open(os.path.join(folder, img_name))
    # Convert from PIL to numpy.
    img_np = np.array(img_pil)
    # Add a new first axis with dimension 1 to stack slices later.
    img_np_exp = img_np[np.newaxis, :]
    imgs.append(img_np_exp)
  # Stack the slices into an image.
  img_np = np.vstack(imgs)
  # Extract contour.
  if contour:
      img_np = (img_np == val).astype(float)
  # Prepare for NIfTI conversion.
  img_np = np.transpose(img_np, (1, 2, 0))
  img_np = np.flip(img_np, axis=0)
  img_np = np.rot90(img_np)
  # Convert to a NIfTI image.
  img_nii = nib.Nifti1Image(img_np, np.eye(4))
  return img_nii

# Main code
for patient_id in tqdm(os.listdir(os.path.join(base_dir, 'Train_Sets', 'MR'))):
  ground = png_to_nifti(os.path.join(base_dir, 'Train_Sets', 'MR', patient_id, 'T1DUAL', 'Ground'))
  nib.save(ground, os.path.join(base_dir, 'labelsTr', f'MR_{patient_id}_T1DUAL_Ground.nii.gz'))
  for phase in os.listdir(os.path.join(base_dir, 'Train_Sets', 'MR', patient_id, 'T1DUAL', 'DICOM_anon')):
    dicom2nifti.dicom_series_to_nifti(os.path.join(base_dir, 'Train_Sets', 'MR', patient_id, 'T1DUAL', 'DICOM_anon', phase),
                                      os.path.join(base_dir, 'imagesTr', f'MR_{patient_id}_T1DUAL_DICOM_anon_{phase}.nii.gz'))
