import os
import sys
import numpy as np
import nibabel as nib

# Get the absolute path to the root directory of your project
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Append the project root to sys.path
sys.path.append(project_root)

from src.nexi.powderaverage.powderaverage import powder_average, normalize_sigma


def test_powder_average():

    powder_average_filename, bval_filename, td_filename = \
        powder_average('tests/nexi/data/phantom.nii.gz',
                       'tests/nexi/data/phantom.bval', 'tests/nexi/data/phantom.td',
                       'tests/nexi/data/mask.nii.gz', 'tests/nexi/data',
                       debug=False)
    assert np.allclose(nib.load(powder_average_filename).get_fdata(),
                       nib.load('tests/nexi/data/powderaverage_dwi_ref.nii.gz').get_fdata(), equal_nan=True)
    assert np.allclose(np.loadtxt(bval_filename), np.loadtxt('tests/nexi/data/powderaverage_ref.bval'))
    assert np.allclose(np.loadtxt(td_filename), np.loadtxt('tests/nexi/data/powderaverage_ref.td'))
    os.remove(powder_average_filename)
    os.remove(bval_filename)
    os.remove(td_filename)


def test_normalize_sigma():
    # sys.path.append('../')
    normalized_sigma_filename = normalize_sigma('tests/nexi/data/phantom.nii.gz',
                                                'tests/nexi/data/lowb_noisemap.nii.gz',
                                                'tests/nexi/data/phantom.bval', 'tests/nexi/data')
    assert np.allclose(nib.load(normalized_sigma_filename).get_fdata(),
                       nib.load('tests/nexi/data/normalized_sigma_ref.nii.gz').get_fdata(), equal_nan=True)
    os.remove(normalized_sigma_filename)
