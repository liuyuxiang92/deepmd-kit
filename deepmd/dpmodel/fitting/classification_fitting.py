# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .invar_fitting import (
    InvarFitting,
)


@InvarFitting.register("classification")
class ClassificationFittingNet(InvarFitting):
    """Fit atomic logits for system-level single-label classification.

    ``num_classes`` is the number of mutually exclusive classes and
    ``label_name`` names the integer class-ID data. Atomic logits are averaged
    over real atoms to produce system logits; softmax belongs to inference or
    the cross-entropy loss, not the fitting network.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        num_classes: int,
        label_name: str = "class",
        neuron: list[int] = [128, 128, 128],
        **kwargs: Any,
    ) -> None:
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise ValueError("num_classes must be an integer of at least 2")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        super().__init__(
            var_name=label_name,
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=num_classes,
            neuron=neuron,
            **kwargs,
        )

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.num_classes],
                    reducible=True,
                    r_differentiable=False,
                    c_differentiable=False,
                    intensive=True,
                ),
                *self._middle_output_def(),
            ]
        )

    @classmethod
    def deserialize(cls, data: dict) -> "ClassificationFittingNet":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("dim_out")
        data["label_name"] = data.pop("var_name")
        return super().deserialize(data)

    def serialize(self) -> dict:
        data = super().serialize()
        data.update(
            {"@version": 1, "type": "classification", "num_classes": self.num_classes}
        )
        return data
