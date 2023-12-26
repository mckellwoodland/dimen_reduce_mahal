"""
Prepares the CHAOS dataset for training.
"""

# Imports
import dicom2nifti
import os
import argparse

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

# Main code
for patient_id in os.listdir(os.path.join(base_dir, 'Train_Sets', 'MR')):
  for phase in os.listdir(os.path.join(base_dir, 'Train_Sets', 'MR', patient_id, 'T1DUAL', 'DICOM_anon')):
    dicom2nifti.dicom_series_to_nifti(os.path.join(base_dir, 'Train_Sets', 'MR', patient_id, 'T1DUAL', 'DICOM_anon', phase),
                                      os.path.join(base_dir, 'imagesTr', f'MR_{patient_id}_T1DUAL_DICOM_anon_{phase}.nii.gz'))
  
