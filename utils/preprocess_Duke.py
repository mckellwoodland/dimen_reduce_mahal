"""
Prepares the Duke Liver MRI dataset for training.
"""

# Imports
import argparse
import dicom2nifti
import os
import tqdm

# Arguments
parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('Required Arguments')
required.add_argument('--base_dir', required=True, type=str, help="Path to directory that contains the '7774566', 'imagesTr', and 'labelsTr' folders.")
args = parser.parse_args()
base_dir = args.base_dir
if not os.path.exists(os.path.join(base_dir, 'amos22')):
    raise FileNotFoundError("The AMOS dataset is not in the given base directory")
if not os.path.exists(os.path.join(base_dir, 'imagesTr')):
    os.mkdir(os.path.join(base_dir, 'imagesTr'))
if not os.path.exists(os.path.join(base_dir, 'labelsTr')):
    os.mkdir(os.path.join(base_dir, 'labelsTr'))

# Functions
def prepend_zeros(num):
    """
    Prepends zeros to a number if it is < 100.
    Input:
        num (int): The number to be prepended.
    Output:
        (str): String with number with preprended zeros.
    """
    num_str = str(num)
    size = len(num_str)
    return "0"*(3-size)+num_str
    
# Main code
index = 0
for patient_id in tqdm.tqdm(sorted(os.listdir(os.path.join(base_dir, '7774566', 'Segmentation')))):
    for img_id in os.listdir(os.path.join(base_dir, '7774566', 'Segmentation', patient_id)):
        dicom2nifti.dicom_series_to_nifti(os.path.join(base_dir, '7774566', 'Segmentation', patient_id, img_id, images),
                                          os.path.join(base_dir, 'imagesTr', f'Liver_{prepend_zeros(index)}_0000.nii.gz'))
        
