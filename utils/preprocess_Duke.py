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
optional = parser.add_argument_group('Optional Arguments')
optional.add_argument('--discard', type=bool, default=True, help='Whether to discard the images with poor quality. Defaults to True.')
args = parser.parse_args()
base_dir = args.base_dir
discard = args.discard
if not os.path.exists(os.path.join(base_dir, 'amos22')):
    raise FileNotFoundError("The AMOS dataset is not in the given base directory")
if not os.path.exists(os.path.join(base_dir, 'imagesTr')):
    os.mkdir(os.path.join(base_dir, 'imagesTr'))
if not os.path.exists(os.path.join(base_dir, 'labelsTr')):
    os.mkdir(os.path.join(base_dir, 'labelsTr'))
if discard:
    discard_patient_ids = [2,3,4,17,20,23,31,32,33,35,38,42,46,50,63,74,75,78,83,84]
    discard_img_ids = {9: [4],
                       10:[3,5],
                       11:[4],
                       18:[3]}
else:
    discard_patient_ids = []
    discard_img_ids = []

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
    for img_id in sorted(os.listdir(os.path.join(base_dir, '7774566', 'Segmentation', patient_id))):
        if int(patient_id) in discard_patient_ids:
            poor_quality = True
        elif int(patient_id) in discard_img_ids.keys():
            if int(img_id) in discard_img_ids[int(patient_id)]:
                poor_quality = True
            else:
                poor_quality = False
        else:
            poor_quality = False
        if not poor_quality:
            dicom2nifti.dicom_series_to_nifti(os.path.join(base_dir, '7774566', 'Segmentation', patient_id, img_id, 'images'),
                                              os.path.join(base_dir, 'imagesTr', f'Liver_{prepend_zeros(index)}_0000.nii.gz'))
        index += 1 
