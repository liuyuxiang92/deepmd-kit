# SPDX-License-Identifier: LGPL-3.0-or-later
"""Classifiers load through DeepEval, including exports with metadata only."""

import zipfile
from pathlib import (
    Path,
)

import numpy as np
import pytest
import torch

from deepmd.infer import (
    DeepClassification,
    DeepEval,
)
from deepmd.pt_expt.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.pt_expt.fitting import (
    ClassificationFittingNet,
)
from deepmd.pt_expt.model import (
    ClassificationModel,
)
from deepmd.pt_expt.utils.serialization import (
    deserialize_to_file,
)


@pytest.fixture(scope="module")
def classifiers(tmp_path_factory):
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
    model = model.to(torch.float64).eval()
    root = tmp_path_factory.mktemp("classification")
    full_path = root / "full.pte"
    metadata_path = root / "metadata.pte"
    deserialize_to_file(str(full_path), {"model": model.serialize()})
    with (
        zipfile.ZipFile(full_path) as original,
        zipfile.ZipFile(metadata_path, "w") as metadata,
    ):
        for entry in original.infolist():
            if not entry.filename.endswith("extra/model.json"):
                metadata.writestr(entry, original.read(entry.filename))
    return full_path, metadata_path


@pytest.mark.parametrize("metadata_only", [False, True])
def test_classifier_export_inference(
    classifiers: tuple[Path, Path], metadata_only
) -> None:
    coords = np.array([[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]])
    types = np.array([0, 0])
    with DeepEval(str(classifiers[int(metadata_only)])) as dp:
        assert isinstance(dp, DeepClassification)
        assert dp.get_num_classes() == 3
        assert dp.get_var_name() == "phase"
        assert dp.get_intensive()
        logits, atomic_logits = dp.eval(coords, None, types, atomic=True)
        assert logits.shape == (1, 3)
        np.testing.assert_allclose(atomic_logits.mean(axis=1), logits)
        probabilities = dp.predict_proba(coords, None, types)
        np.testing.assert_allclose(probabilities.sum(axis=-1), 1)
        np.testing.assert_array_equal(
            dp.predict(coords, None, types), logits.argmax(-1)
        )
    with DeepEval(str(classifiers[0])) as full:
        np.testing.assert_allclose(full.eval(coords, None, types)[0], logits)
