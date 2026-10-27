# SPDX-License-Identifier: LGPL-3.0-or-later
"""Classifier prediction helpers and frame-weighted ``dp test`` metrics."""

from types import (
    SimpleNamespace,
)
from unittest.mock import (
    Mock,
)

import numpy as np
import pytest

from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting import (
    ClassificationFittingNet,
)
from deepmd.dpmodel.model import (
    ClassificationModel,
)
from deepmd.dpmodel.utils.serialization import (
    save_dp_model,
)
from deepmd.infer import (
    DeepClassification,
    DeepEval,
)
from deepmd.infer.model_test import (
    ClassificationTester,
    build_tester,
)
from deepmd.utils.weight_avg import (
    merge_weighted_errors,
    weighted_average,
)


def _evaluator(logits: np.ndarray) -> DeepClassification:
    dp = object.__new__(DeepClassification)
    dp.deep_eval = SimpleNamespace(
        get_task_dim=lambda: logits.shape[1],
        get_var_name=lambda: "phase",
        get_dim_fparam=lambda: 0,
        get_dim_aparam=lambda: 0,
    )
    dp.eval = Mock(return_value=(logits,))
    return dp


def test_prediction_helpers_preserve_raw_logits() -> None:
    logits = np.array([[1000.0, 1002.0, 1001.0], [-1000.0, -1001.0, -999.0]])
    dp = _evaluator(logits)
    assert dp.get_num_classes() == 3
    np.testing.assert_array_equal(dp.predict(None, None, None), [1, 2])
    probabilities = dp.predict_proba(None, None, None)
    np.testing.assert_allclose(probabilities.sum(axis=-1), np.ones(2))
    np.testing.assert_array_equal(np.argmax(probabilities, axis=-1), [1, 2])
    np.testing.assert_array_equal(dp.eval()[0], logits)


def test_native_dpmodel_classifier_inference(tmp_path) -> None:
    descriptor = DescrptSeA(3.0, 2.5, [4], neuron=[4, 8], axis_neuron=2)
    fitting = ClassificationFittingNet(
        1,
        descriptor.get_dim_out(),
        num_classes=3,
        label_name="phase",
        neuron=[4],
        mixed_types=descriptor.mixed_types(),
        seed=7,
    )
    model = ClassificationModel(descriptor, fitting, type_map=["H"])
    model_file = tmp_path / "classifier.dp"
    save_dp_model(str(model_file), {"model": model.serialize()})
    coords = np.array([[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]])
    types = np.array([0, 0])
    with DeepEval(str(model_file)) as dp:
        assert isinstance(dp, DeepClassification)
        logits, atomic = dp.eval(coords, None, types, atomic=True)
        np.testing.assert_allclose(atomic.mean(axis=1), logits)
        np.testing.assert_allclose(dp.predict_proba(coords, None, types).sum(-1), 1)


def _chunk(labels: list[float], natoms: int = 2) -> dict:
    nframes = len(labels)
    return {
        "coord": np.zeros((nframes, natoms * 3)),
        "box": np.tile(np.eye(3).reshape(1, 9), (nframes, 1)),
        "type": np.zeros((nframes, natoms), dtype=np.int32),
        "phase": np.asarray(labels).reshape(-1, 1),
    }


def test_chunked_testing_weights_by_frames_and_writes_classes(tmp_path) -> None:
    logits = np.array([[1000.0, 999.0], [-1000.0, -1001.0]])
    dp = _evaluator(logits)
    dp.eval.side_effect = [(logits[:1],), (logits,)]
    chunks = [_chunk([0], natoms=2), _chunk([1, 1], natoms=5)]
    data = SimpleNamespace(
        pbc=False,
        mixed_type=False,
        add=Mock(),
        iter_test=lambda **kwargs: iter(chunks),
    )
    tester = build_tester(dp, atomic=True)
    assert isinstance(tester, ClassificationTester)
    result = tester.run(data, "classes", 3, str(tmp_path / "details"))
    data.add.assert_called_once_with(
        "phase", 1, atomic=False, must=True, high_prec=True
    )
    assert result["accuracy"] == pytest.approx((1 / 3, 3))
    assert result["cross_entropy"] == pytest.approx((np.log1p(np.exp(-1)) + 2 / 3, 3))
    np.testing.assert_array_equal(
        np.loadtxt(tmp_path / "details.classification.out.2"), [1, 0]
    )


@pytest.mark.parametrize("label", [0.5, np.nan, np.inf, -1, 2])
def test_testing_rejects_invalid_labels(label: float) -> None:
    dp = _evaluator(np.zeros((1, 2)))
    data = SimpleNamespace(
        pbc=False,
        mixed_type=False,
        add=Mock(),
        iter_test=lambda **kwargs: iter([_chunk([label])]),
    )
    with pytest.raises(ValueError, match="Class labels must"):
        build_tester(dp, atomic=False).run(data, "invalid", 1, None)


def test_weighted_metrics_preserve_accuracy_and_rmse_rules() -> None:
    errors = [
        {"accuracy": (1.0, 1), "cross_entropy": (2.0, 1), "rmse_f": (2.0, 1)},
        {"accuracy": (0.0, 3), "cross_entropy": (4.0, 3), "rmse_f": (4.0, 3)},
    ]
    expected = {"accuracy": 0.25, "cross_entropy": 3.5, "rmse_f": np.sqrt(13.0)}
    assert weighted_average(errors) == pytest.approx(expected)
    merged = merge_weighted_errors(errors)
    assert weighted_average([merged]) == pytest.approx(expected)
    assert all(weight == 4 for _, weight in merged.values())
