# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.fitting.classification_fitting import (
    ClassificationFittingNet,
)
from deepmd.utils.path import (
    DPPath,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPClassificationAtomicModel(DPAtomicModel):
    """Atomic classification logits, without categorical-label statistics."""

    def __init__(
        self, descriptor: Any, fitting: Any, type_map: list[str], **kwargs: Any
    ) -> None:
        if not isinstance(fitting, ClassificationFittingNet):
            raise TypeError("fitting must be a ClassificationFittingNet")
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def get_intensive(self) -> bool:
        return True

    def compute_or_load_out_stat(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        stat_file_path: DPPath | None = None,
    ) -> None:
        """Class IDs are categorical and have no regression statistics."""

    def change_out_bias(
        self,
        sample_merged: Callable[[], list[dict]] | list[dict],
        stat_file_path: DPPath | None = None,
        bias_adjust_mode: str = "change-by-statistic",
        observed_type: list[str] | None = None,
        preset_out_bias: dict | None = None,
    ) -> None:
        """Keep the learned logits independent of label-statistic bias changes."""

    def apply_out_stat(self, ret: dict[str, Array], atype: Array) -> dict[str, Array]:
        """Return logits without scaling or shifting by label statistics."""
        return ret
