# nexi

[![PyPI - Version](https://img.shields.io/pypi/v/nexi.svg)](https://pypi.org/project/nexi)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nexi.svg)](https://pypi.org/project/nexi)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install nexi
```

## Prerequisites

### Preprocessing

Before proceeding, make sure to preprocess your data with the following steps:
- Marchenko-Pastur principal component analysis (MP-PCA) denoising ([Veraart et al., 2016](https://doi.org/10.1016/j.neuroimage.2016.08.016)). Recommended algorithm : [dwidenoise from mrtrix](https://mrtrix.readthedocs.io/en/dev/reference/commands/dwidenoise.html)
- Gibbs ringing correction ([Kellner et al., 2016](https://doi.org/10.1002/mrm.26054)). Recommended algorithm : [FSL implementation](https://bitbucket.org/reisert/unring/src/master/)
- Distortion correction using FSL topup ([Andersson et al., 2003](https://doi.org/10.1002/mrm.10335), [Andersson et al., 2016](https://doi.org/10.1016/j.neuroimage.2015.10.019)). Recommended algorithm : [FSL topup](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup)
- Eddy current and motion correction ([Andersson and Sotiropoulos, 2016](https://doi.org/10.1016/j.neuroimage.2015.12.037)). Recommended algorithm : [FSL eddy](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy)

Additionally, you need to compute another noisemap using only the small b-values (b < 2 ms/µm²) and MP-PCA. This noisemap will be used to calculate the signal-to-noise ratio (SNR) of the data.

Furthermore, you can provide a mask of grey matter tissue if available. This mask can be used to restrict the processing to specific regions of interest. If a mask is not provided, the algorithms will be applied to the entire image, voxel by voxel, as long as there are no NaN values present.

To compute a grey matter mask, one common approach involves using a T1 image, [FastSurfer](https://deep-mi.org/research/fastsurfer/), and performing registration to the diffusion (b = 0 ms/µm²) space. However, you can choose any other method to compute a grey matter mask.

## Usage

### Estimate NEXI parameters

```
estimate_nexi(dwi_path, bvals_path, td_path, lowb_noisemap_path, out_path)
```

## License

`nexi` is distributed under the terms of the [Apache License 2.0](https://spdx.org/licenses/Apache-2.0.html).
