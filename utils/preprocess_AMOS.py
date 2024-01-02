"""
Prepares the AMOS dataset for training.
Default range values focuses on the liver.
"""

# Imports
import argparse
import os

import dicom2nifti
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

# Arguments
parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('Required Arguments')
required.add_argument('--base_dir', required=True, type=str, help="Path to directory that contains the 'amos22', 'imagesTr', and 'labelsTr' folders.")
optional = parser.add_argument_group('Optional Arguments')
optional.add_argument('--val', type=int, default=6, help="Voxel value that belongs to the contour to be extracted. Defaults to 6 (liver).")
args = parser.parse_args()
base_dir = args.base_dir
val = args.val
if not os.path.exists(os.path.join(base_dir, 'amos22')):
  raise FileNotFoundError("The AMOS dataset is not in the given base directory")
if not os.path.exists(os.path.join(base_dir, 'imagesTr')):
  os.mkdir(os.path.join(base_dir, 'imagesTr'))
if not os.path.exists(os.path.join(base_dir, 'labelsTr')):
  os.mkdir(os.path.join(base_dir, 'labelsTr'))

# Main code
for folder in ['imagesTr', 'imagesVa']:
