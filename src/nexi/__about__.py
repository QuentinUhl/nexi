# SPDX-FileCopyrightText: 2023-present Quentin Uhl <quentin.uhl@wanadoo.fr>
#
# SPDX-License-Identifier: MIT
__version__ = "1.0"
if __name__ == '__main__':
    from .estimate_nexi_noiseless import estimate_nexi_noiseless
    from .estimate_nexi import estimate_nexi
    from .powderaverage.powderaverage import powder_average, save_data_as_npz, normalize_sigma
