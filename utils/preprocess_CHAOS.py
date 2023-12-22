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
required.add_argument('--base_dir', required=True, type=str, help="Path to directory that contains the 'Train_Sets' folder and will contain the 'imagesTr' and 'labelsTr' folders")
args = parser.parse_args()
base_dir = args.base_dir
if not os.path.exists(os.path.join(base_dir, 'Train_Sets')):
  raise FileNotFoundError("The CHAOS dataset is not in the given base directory")
if not os.path.exists(os.path.join(base_dir, 'imagesTr')):
  os.mkdir(os.path.join(base_dir, 'imagesTr'))
if not os.path.exists(os.path.join(base_dir, 'labelsTr')):
  os.mkdir(os.path.join(base_dir, 'labelsTr'))


  
