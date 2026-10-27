# SPDX-License-Identifier: LGPL-3.0-or-later
"""Class-label validation and numerically stable system classification loss."""

import numpy as np
import pytest

from deepmd.dpmodel.loss import (
    ClassificationLoss,
)
from deepmd.dpmodel.loss.loss import (
    Loss,
)


@pytest.mark.parametrize("column_labels", [False, True])
def test_cross_entropy_and_accuracy(column_labels: bool) -> None:
    """A common large logit offset must cancel without losing precision."""
    logits = np.array([[1000.0, 1001.0, 1002.0], [-1001.0, -1000.0, -999.0]])
    labels = np.array([2, 1])
    if column_labels:
        labels = labels[:, None]
    objective = ClassificationLoss(3, "phase")
    loss, metrics = objective(1e-3, 7, {"phase": logits}, {"phase": labels})
    expected = np.log(np.exp(-2.0) + np.exp(-1.0) + 1.0) + 0.5
    np.testing.assert_allclose(loss, expected, rtol=1e-14)
    assert metrics == {"accuracy": 0.5}


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (np.array([0.5, 1.0]), "must be integers"),
        (np.array([np.nan, 1.0]), "must be finite integers"),
        (np.array([np.inf, 1.0]), "must be finite integers"),
        (np.array([-1, 1]), r"must be in \[0, 3\)"),
        (np.array([0, 3]), r"must be in \[0, 3\)"),
        (np.array([2**64 - 1, 1], dtype=np.uint64), r"must be in \[0, 3\)"),
        (np.array([0j, 1j]), "must be integers"),
        (np.array([True, False]), "must be integers"),
        (np.zeros((2, 3)), "Expected labels with shape"),
        (np.zeros(3), "Expected labels with shape"),
        (np.zeros((2, 1, 1)), "Expected labels with shape"),
    ],
)
def test_invalid_labels(labels: np.ndarray, message: str) -> None:
    objective = ClassificationLoss(3, "phase")
    with pytest.raises(ValueError, match=message):
        objective(0.0, 1, {"phase": np.zeros((2, 3))}, {"phase": labels})


@pytest.mark.parametrize("shape", [(2, 2), (2, 1, 3), (3,)])
def test_invalid_logits(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="Expected logits with shape"):
        ClassificationLoss(3, "phase")(
            0.0, 1, {"phase": np.zeros(shape)}, {"phase": np.zeros(2)}
        )


def test_empty_batch() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        ClassificationLoss(3, "phase")(
            0.0, 1, {"phase": np.zeros((0, 3))}, {"phase": np.zeros(0)}
        )


@pytest.mark.parametrize("num_classes", [0, 1, 2.5, True])
def test_invalid_class_count(num_classes) -> None:
    with pytest.raises(ValueError, match="num_classes must be an integer"):
        ClassificationLoss(num_classes, "phase")


def test_serialization_registration_and_data_contract() -> None:
    objective = ClassificationLoss(3, "phase")
    assert Loss.get_class_by_type("classification") is ClassificationLoss
    assert ClassificationLoss.deserialize(objective.serialize()).serialize() == (
        objective.serialize()
    )
    assert objective.training_metric_names == ("accuracy",)
    assert objective.supports_ragged_batches
    requirement = objective.label_requirement[0]
    assert requirement.key == "phase"
    assert requirement.ndof == 1
    assert requirement.must
    assert not requirement.atomic


def test_array_api_strict() -> None:
    xp = pytest.importorskip("array_api_strict")
    logits = xp.asarray([[1001.0, 1000.0], [-1000.0, -1001.0]])
    labels = xp.asarray([0.0, 1.0])
    loss, metrics = ClassificationLoss(2, "phase")(
        0.0, 1, {"phase": logits}, {"phase": labels}
    )
    np.testing.assert_allclose(float(loss), np.log1p(np.exp(-1.0)) + 0.5)
    assert float(metrics["accuracy"]) == 0.5
