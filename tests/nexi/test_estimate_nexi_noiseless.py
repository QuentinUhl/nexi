import os
import numpy as np
import nibabel as nib
from src.nexi.estimate_nexi_noiseless import estimate_nexi_noiseless


def test_estimate_nexi_noiseless():
    np.random.seed(123)
    if not os.path.isdir('tests/nexi/data/nexi'):
        os.mkdir('tests/nexi/data/nexi')
    estimate_nexi_noiseless('tests/nexi/data/phantom.nii.gz',
                            'tests/nexi/data/phantom.bval', 'tests/nexi/data/phantom.td',
                            'tests/nexi/data/nexi', mask_path='tests/nexi/data/mask.nii.gz',
                            debug=False)

    powder_average_filename = 'tests/nexi/data/nexi/powderaverage_dwi.nii.gz'
    bval_filename = 'tests/nexi/data/nexi/powderaverage.bval'
    td_filename = 'tests/nexi/data/nexi/powderaverage.td'
    powder_average_npz_filename = 'tests/nexi/data/nexi/powderaverage_signal.npz'

    assert np.allclose(nib.load(powder_average_filename).get_fdata(),
                       nib.load('tests/nexi/data/powderaverage_dwi_ref.nii.gz').get_fdata(), equal_nan=True)
    assert np.allclose(np.loadtxt(bval_filename), np.loadtxt('tests/nexi/data/powderaverage_ref.bval'))
    assert np.allclose(np.loadtxt(td_filename), np.loadtxt('tests/nexi/data/powderaverage_ref.td'))

    os.remove(powder_average_filename)
    os.remove(bval_filename)
    os.remove(td_filename)
    os.remove(powder_average_npz_filename)

    parameters = ["t_ex", "di", "de", "f"]
    for param in parameters:
        param_filename = f'tests/nexi/data/nexi/nexi_{param}.nii.gz'
        param_ref_filename = f'tests/nexi/data/models_ref/nexi_{param}_ref.nii.gz'
        assert np.allclose(nib.load(param_filename).get_fdata(),
                           nib.load(param_ref_filename).get_fdata(), equal_nan=True)
        os.remove(param_filename)

    os.rmdir('tests/nexi/data/nexi')
