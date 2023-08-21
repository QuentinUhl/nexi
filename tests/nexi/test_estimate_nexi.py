import os
import sys
import numpy as np
import nibabel as nib

# Get the absolute path to the root directory of your project
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Append the project root to sys.path
sys.path.append(project_root)

from src.nexi.estimate_nexi import estimate_nexi


def test_estimate_nexi():
    np.random.seed(123)
    if not os.path.isdir('tests/nexi/data/nexi_rice_mean'):
        os.mkdir('tests/nexi/data/nexi_rice_mean')
    estimate_nexi(
        'tests/nexi/data/phantom.nii.gz',
        'tests/nexi/data/phantom.bval',
        'tests/nexi/data/phantom.td',
        'tests/nexi/data/lowb_noisemap.nii.gz',
        'tests/nexi/data/nexi_rice_mean',
        mask_path='tests/nexi/data/mask.nii.gz',
        debug=False,
    )

    powder_average_filename = 'tests/nexi/data/nexi_rice_mean/powderaverage_dwi.nii.gz'
    bval_filename = 'tests/nexi/data/nexi_rice_mean/powderaverage.bval'
    td_filename = 'tests/nexi/data/nexi_rice_mean/powderaverage.td'
    powder_average_npz_filename = 'tests/nexi/data/nexi_rice_mean/powderaverage_signal.npz'
    sigma_filename = 'tests/nexi/data/nexi_rice_mean/normalized_sigma.nii.gz'

    assert np.allclose(
        nib.load(powder_average_filename).get_fdata(),
        nib.load('tests/nexi/data/powderaverage_dwi_ref.nii.gz').get_fdata(),
        equal_nan=True,
    )
    assert np.allclose(np.loadtxt(bval_filename), np.loadtxt('tests/nexi/data/powderaverage_ref.bval'))
    assert np.allclose(np.loadtxt(td_filename), np.loadtxt('tests/nexi/data/powderaverage_ref.td'))

    os.remove(powder_average_filename)
    os.remove(bval_filename)
    os.remove(td_filename)
    os.remove(powder_average_npz_filename)
    os.remove(sigma_filename)

    parameters = ["t_ex", "di", "de", "f", "sigma"]
    for param in parameters:
        param_filename = f'tests/nexi/data/nexi_rice_mean/nexi_rice_mean_{param}.nii.gz'
        param_ref_filename = f'tests/nexi/data/models_ref/nexi_rice_mean_{param}_ref.nii.gz'
        assert np.allclose(
            nib.load(param_filename).get_fdata(), nib.load(param_ref_filename).get_fdata(), equal_nan=True
        )
        os.remove(param_filename)

    os.rmdir('tests/nexi/data/nexi_rice_mean')
