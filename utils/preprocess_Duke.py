"""
Prepares the Duke Liver MRI dataset for training.
"""

# Imports
import argparse
import os

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

# Main code
