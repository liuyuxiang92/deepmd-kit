# SPDX-License-Identifier: LGPL-3.0-or-later
"""Cross-entropy for one class label per system."""

from numbers import (
    Integral,
)
from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
    xp_take_along_axis,
)
from deepmd.dpmodel.loss.loss import (
    Loss,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


@Loss.register("classification")
class ClassificationLoss(Loss):
    """Mean cross-entropy from raw system logits and integer class labels.

    Parameters
    ----------
    num_classes : int
        Number of classes. Labels must be in ``[0, num_classes)``.
    var_name : str
        Name of both the model logits and the scalar data label.
    """

    def __init__(self, num_classes: int, var_name: str, **kwargs: Any) -> None:
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, Integral)
            or num_classes < 2
        ):
            raise ValueError(
                "num_classes must be an integer greater than or equal to 2"
            )
        self.num_classes = int(num_classes)
        self.var_name = var_name

    def call(
        self,
        learning_rate: float,
        natoms: int,
        model_dict: dict[str, Array],
        label_dict: dict[str, Array],
        mae: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        """Evaluate cross-entropy and the fraction of correctly classified frames."""
        del learning_rate, natoms, mae
        logits = model_dict[self.var_name]
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected logits with shape (batch, {self.num_classes}), "
                f"but got {tuple(logits.shape)}"
            )
        if logits.shape[0] == 0:
            raise ValueError("Classification requires at least one frame")
        xp = array_api_compat.array_namespace(logits)
        target = xp.asarray(
            label_dict[self.var_name], device=array_api_compat.device(logits)
        )
        if target.ndim == 2 and target.shape[1] == 1:
            target = xp.squeeze(target, axis=1)
        if target.ndim != 1 or target.shape[0] != logits.shape[0]:
            raise ValueError(
                f"Expected labels with shape ({logits.shape[0]},) or "
                f"({logits.shape[0]}, 1), but got {tuple(target.shape)}"
            )
        if xp.isdtype(target.dtype, "real floating"):
            if not bool(xp.all(xp.isfinite(target))):
                raise ValueError("Class labels must be finite integers")
            if not bool(xp.all(target == xp.floor(target))):
                raise ValueError("Class labels must be integers")
        elif not xp.isdtype(target.dtype, "integral"):
            raise ValueError("Class labels must be integers")
        # Validate before casting: an out-of-range unsigned or floating-point
        # label must not wrap or truncate into a valid class index.
        if bool(xp.any(target < 0)) or bool(xp.any(target >= self.num_classes)):
            raise ValueError(f"Class labels must be in [0, {self.num_classes})")
        target = xp.astype(target, xp.int64)

        # Subtract the maximum before exponentiation, and gather from the
        # shifted logits to avoid cancellation for large common offsets.
        shifted = logits - xp.max(logits, axis=-1, keepdims=True)
        log_normalizer = xp.log(xp.sum(xp.exp(shifted), axis=-1))
        selected = xp_take_along_axis(shifted, target[:, None], axis=1)[:, 0]
        loss = xp.mean(log_normalizer - selected)
        accuracy = xp.mean(
            xp.astype(xp.argmax(logits, axis=-1) == target, logits.dtype)
        )
        return loss, {"accuracy": accuracy}

    @property
    def training_metric_names(self) -> tuple[str, ...]:
        """Return display metrics without applying any RMSE transformation."""
        return ("accuracy",)

    @property
    def supports_ragged_batches(self) -> bool:
        """System logits keep one row per frame for both dense and ragged atoms."""
        return True

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Require one scalar class index per frame."""
        return [
            DataRequirementItem(
                self.var_name,
                ndof=1,
                atomic=False,
                must=True,
            )
        ]

    def serialize(self) -> dict:
        """Serialize the loss module."""
        return {
            "@class": "ClassificationLoss",
            "@version": 1,
            "num_classes": self.num_classes,
            "var_name": self.var_name,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "ClassificationLoss":
        """Deserialize the loss module."""
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        return cls(**data)
