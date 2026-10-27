# SPDX-License-Identifier: LGPL-3.0-or-later
"""Gradient and training integration checks for classification."""

from types import (
    SimpleNamespace,
)
from unittest.mock import (
    Mock,
)

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from deepmd.dpmodel.train import (
    TrainingMetricAccumulator,
    TrainingTask,
)
from deepmd.pt_expt.loss import (
    ClassificationLoss,
)
from deepmd.pt_expt.train.training import (
    Trainer,
    get_loss,
)
from deepmd.pt_expt.utils import (
    env,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cross_entropy_value_and_gradient(dtype: torch.dtype) -> None:
    logits = torch.tensor(
        [[-1000, -999, -1001], [1001, 1000, 1000], [0, -1, 1]],
        dtype=dtype,
        device=env.DEVICE,
        requires_grad=True,
    )
    target = torch.tensor([1, 1, 0], device=env.DEVICE)
    reference = F.cross_entropy(logits, target)
    expected_gradient = torch.autograd.grad(reference, logits)[0]
    loss, metrics = ClassificationLoss(3, "phase")(
        1e-3,
        7,
        {"phase": logits},
        {"phase": target.to(dtype).reshape(-1, 1)},
    )
    actual_gradient = torch.autograd.grad(loss, logits)[0]
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert metrics["accuracy"].item() == pytest.approx(1 / 3)
    assert not metrics["accuracy"].requires_grad


@pytest.mark.parametrize("label", [0.5, float("nan"), float("inf"), -1, 3])
def test_invalid_tensor_labels(label: float) -> None:
    logits = torch.zeros((1, 3), dtype=torch.float64, device=env.DEVICE)
    target = torch.tensor([label], dtype=torch.float64, device=env.DEVICE)
    with pytest.raises(ValueError, match="Class labels must"):
        ClassificationLoss(3, "phase")(0.0, 1, {"phase": logits}, {"phase": target})


def test_factory_and_training_metric_values() -> None:
    model = SimpleNamespace(get_num_classes=lambda: 3, get_var_name=lambda: "phase")
    objective = get_loss({"type": "classification"}, 1e-3, 2, model)
    assert isinstance(objective, ClassificationLoss)
    assert objective.num_classes == 3
    assert objective.var_name == "phase"
    metrics = {"accuracy": torch.tensor(0.25)}
    assert Trainer._more_loss_to_float(metrics) == {"accuracy": 0.25}
    accumulator = TrainingMetricAccumulator(
        {"Default": objective.training_metric_names}
    )
    accumulator.add("Default", metrics)
    assert accumulator.average("Default") == {"accuracy": 0.25}


@pytest.mark.parametrize("ragged", [False, True])
def test_validation_accuracy_counts_frames(ragged: bool) -> None:
    """One successful small frame plus two failing large frames gives 1/3."""
    trainer = Trainer.__new__(Trainer)
    trainer.sharding = SimpleNamespace(shards_parameters=False)
    trainer.losses = {"Default": ClassificationLoss(3, "phase")}
    trainer._display_cur_lr_sched = 1e-3
    first_types = torch.zeros((1, 2), dtype=torch.int64)
    second_types = torch.zeros((2, 5), dtype=torch.int64)
    if ragged:
        first_types = first_types.flatten()
        second_types = second_types.flatten()
    trainer.get_data = Mock(
        side_effect=[
            ({"atype": first_types}, {"phase": torch.tensor([[0]])}),
            ({"atype": second_types}, {"phase": torch.tensor([[1], [2]])}),
        ]
    )
    trainer.wrapper = Mock(spec=torch.nn.Module)
    trainer.wrapper.side_effect = [
        ({}, None, {"accuracy": torch.tensor(1.0)}),
        ({}, None, {"accuracy": torch.tensor(0.0)}),
    ]
    task = TrainingTask("Default", None, validation_data=object(), valid_numb_batch=2)
    metrics = trainer.evaluate_validation(task, 0, None)
    np.testing.assert_allclose(metrics["accuracy"], 1 / 3)
