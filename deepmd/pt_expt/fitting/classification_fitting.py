# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.fitting.classification_fitting import (
    ClassificationFittingNet as ClassificationFittingNetDP,
)
from deepmd.pt_expt.common import (
    torch_module,
)

from .base_fitting import (
    BaseFitting,
)


@BaseFitting.register("classification")
@torch_module
class ClassificationFittingNet(ClassificationFittingNetDP):
    pass
