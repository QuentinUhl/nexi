# Neurite Exchange Imaging (NEXI) model estimator for gray matter diffusion MRI

[![PyPI - Version](https://img.shields.io/pypi/v/nexi.svg)](https://pypi.org/project/nexi)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/nexi)](#)
[![GitHub](https://img.shields.io/github/license/QuentinUhl/nexi)](#)
[![GitHub top language](https://img.shields.io/github/languages/top/QuentinUhl/nexi?color=lightgray)](#)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nexi.svg)](https://pypi.org/project/nexi)

-----

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Prerequisites](#prerequisites)
- [Citation](#citation)
- [License](#license)

## Installation

```console
pip install nexi
```

## Usage

### Estimate NEXI parameters

To estimate NEXI parameters using the nexi package, you can use the estimate_nexi function. This function takes several parameters that you need to provide in order to perform the estimation accurately.

```
estimate_nexi(dwi_path, bvals_path, td_path, lowb_noisemap_path, out_path)
```

`dwi_path`: The path to the diffusion-weighted image (DWI) data in NIfTI format. This data contains the preprocessed diffusion-weighted volumes acquired from your imaging study.

`bvals_path`: The path to the b-values file corresponding to the DWI data. B-values specify the strength and timing of diffusion sensitization gradients for each volume in the DWI data.

`td_path`: The path to the diffusion time (td) file, also known as Δ. This file provides information about the diffusion time for each volume in the DWI data. The diffusion time is the time between the two gradient pulses. 

`lowb_noisemap_path`: The path to the noisemap calculated using only the small b-values (b < 2 ms/µm²) and Marchenko-Pastur principal component analysis (MP-PCA) denoising. This noisemap is used to calculate the signal-to-noise ratio (SNR) of the data.

`out_path`: The folder where the estimated NEXI parameters will be saved as output.

## Prerequisites

### Data Acquisition

For accurate NEXI parameter estimation using the nexi package, acquire PGSE EPI (Pulsed Gradient Spin Echo Echo-Planar Imaging) diffusion MRI data with diverse combinations of b values and diffusion times. Ensure reasonable signal-to-noise ratio (SNR) in the data for accurate parameter estimation.

### Preprocessing

Before proceeding, make sure to preprocess your data with the following steps:
- Marchenko-Pastur principal component analysis (MP-PCA) denoising ([Veraart et al., 2016](https://doi.org/10.1016/j.neuroimage.2016.08.016)). Recommended algorithm : [dwidenoise from mrtrix](https://mrtrix.readthedocs.io/en/dev/reference/commands/dwidenoise.html)
- Gibbs ringing correction ([Kellner et al., 2016](https://doi.org/10.1002/mrm.26054)). Recommended algorithm : [FSL implementation](https://bitbucket.org/reisert/unring/src/master/)
- Distortion correction using FSL topup ([Andersson et al., 2003](https://doi.org/10.1002/mrm.10335), [Andersson et al., 2016](https://doi.org/10.1016/j.neuroimage.2015.10.019)). Recommended algorithm : [FSL topup](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup)
- Eddy current and motion correction ([Andersson and Sotiropoulos, 2016](https://doi.org/10.1016/j.neuroimage.2015.12.037)). Recommended algorithm : [FSL eddy](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy)

Additionally, you need to compute another noisemap using only the small b-values (b < 2 ms/µm²) and MP-PCA. This noisemap will be used to calculate the signal-to-noise ratio (SNR) of the data.

Furthermore, you can provide a mask of grey matter tissue if available. This mask can be used to restrict the processing to specific regions of interest. If a mask is not provided, the algorithms will be applied to the entire image, voxel by voxel, as long as there are no NaN values present.

To compute a grey matter mask, one common approach involves using a T1 image, [FastSurfer](https://deep-mi.org/research/fastsurfer/), and performing registration to the diffusion (b = 0 ms/µm²) space. However, you can choose any other method to compute a grey matter mask.

## Citation

If you use this package in your research, please consider citing the following papers:

### Original NEXI Paper

Ileana O. Jelescu, Alexandre de Skowronski, Françoise Geffroy, Marco Palombo, Dmitry S. Novikov, [Neurite Exchange Imaging (NEXI): A minimal model of diffusion in gray matter with inter-compartment water exchange](https://www.sciencedirect.com/science/article/pii/S1053811922003986), NeuroImage, 2022.

### First application on human gray matter / Development of this package

Quentin Uhl, Tommaso Pavan, Malwina Molendowska, Derek K. Jones, Marco Palombo, Ileana O. Jelescu, [Quantifying human gray matter microstructure using NEXI and 300 mT/m gradients](https://arxiv.org/abs/2307.09492), Arxiv, 2023.


## License

`nexi` is distributed under the terms of the [Apache License 2.0](https://spdx.org/licenses/Apache-2.0.html).
