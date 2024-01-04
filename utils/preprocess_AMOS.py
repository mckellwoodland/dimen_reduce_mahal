"""
Prepares the AMOS dataset for training.
Default range values focuses on the liver.
"""

# Imports
import argparse
import shutil
import os
import tqdm
import nibabel as nib

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

# Functions
def extract_contour(img, val):
    """
    creates a binary mask for one organ of interest.

    Inputs:
        img (ndarray): 3D array.
        val (int): Value that pertains to the organ contour of interest.

    Output:
        (ndarray): Binary mask.
    """
    return (img == val).astype(float)

# Main code
for folder in ['Tr', 'Va']:
    for f in tqdm.tqdm(os.listdir(os.path.join(base_dir, 'amos22', f'images{folder}'))):
        if '.nii.gz' in f:
            num = int(f.split('_')[-1].split('.')[0])
            if (num >= 500) and (num <= 600):
                shutil.copy(os.path.join(base_dir, 'amos22', f'images{folder}', f), os.path.join(base_dir, 'imagesTr', f))
                img_nib = nib.load(os.path.join(base_dir, 'amos22', f'labels{folder}', f))
                img = img_nib.get_fdata()
                img = extract_contour(img, val)
                img_nib = nib.Nifti1Image(img, img_nib.affine, img_nib.header)
                nib.save(img_nib, os.path.join(base_dir, 'labelsTr', f))
