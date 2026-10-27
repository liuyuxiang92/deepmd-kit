# SPDX-License-Identifier: LGPL-3.0-or-later
import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from deepmd.dpmodel.fitting.classification_fitting import (
    ClassificationFittingNet,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel as DPBaseModel,
)
from deepmd.infer import (
    DeepClassification,
    DeepEval,
)
from deepmd.pt_expt.model import (
    BaseModel,
    get_model,
)
from deepmd.pt_expt.model.graph_lower import (
    model_uses_graph_lower,
)
from deepmd.pt_expt.utils.serialization import (
    deserialize_to_file,
)

from ...dpa4_fixtures import (
    jitter_zero_arrays,
)


def _config() -> dict:
    return {
        "type": "dpa4",
        "type_map": ["O", "H"],
        "descriptor": {
            "sel": 20,
            "rcut": 4.0,
            "channels": 8,
            "n_radial": 4,
            "lmax": 1,
            "mmax": 1,
            "n_blocks": 1,
            "precision": "float64",
        },
        "fitting_net": {
            "type": "classification",
            "num_classes": 3,
            "label_name": "phase",
            "neuron": [8],
            "precision": "float64",
        },
    }


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]],
            dtype=torch.float64,
        ),
        torch.tensor([[0, 1, 1]], dtype=torch.int64),
    )


@pytest.mark.parametrize("num_classes", [1, 0, -1, 2.5, True])
def test_invalid_num_classes(num_classes) -> None:
    with pytest.raises(ValueError, match="num_classes"):
        ClassificationFittingNet(2, 8, num_classes)


def test_dpa4_classification_logits_and_gradients() -> None:
    config = _config()
    original = copy.deepcopy(config)
    model = get_model(config).cpu()
    coord, atype = _inputs()
    prediction = model(coord, atype)
    assert config == original
    assert model.get_num_classes() == model.get_task_dim() == 3
    assert model.get_var_name() == "phase"
    assert model.get_intensive()
    assert prediction["phase"].shape == (1, 3)
    torch.testing.assert_close(
        prediction["phase"], prediction["atom_phase"].mean(dim=1)
    )
    F.cross_entropy(prediction["phase"], torch.tensor([2])).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    assert any(torch.count_nonzero(grad) for grad in gradients)


@pytest.mark.parametrize(
    "neuron,numb_fparam,numb_aparam,dim_case_embd,expected",
    [([0], 0, 0, 0, [32]), ([0, 16, 0], 2, 3, 1, [64, 16, 64])],
)
def test_dpa4_classification_auto_width_matches_pt(
    neuron, numb_fparam, numb_aparam, dim_case_embd, expected
) -> None:
    from deepmd.pt.model.model import get_model as get_pt_model

    config = _config()
    config["fitting_net"].update(
        neuron=neuron,
        numb_fparam=numb_fparam,
        numb_aparam=numb_aparam,
        dim_case_embd=dim_case_embd,
    )
    model = get_model(config).cpu()
    reference = get_pt_model(config).cpu()
    assert model.get_fitting_net().neuron == expected
    assert reference.get_fitting_net().neuron == expected
    # The caller's reusable training configuration keeps the auto markers.
    assert config["fitting_net"]["neuron"] == neuron


def test_classification_skips_categorical_output_statistics() -> None:
    model = get_model(_config()).cpu()
    coord, atype = _inputs()
    before = model(coord, atype)["phase"]

    def forbidden_sampler():
        raise AssertionError("classification must not sample label statistics")

    atomic_model = model.atomic_model
    atomic_model.compute_or_load_out_stat(forbidden_sampler)
    atomic_model.change_out_bias(forbidden_sampler, observed_type=["O", "H"])
    with torch.no_grad():
        atomic_model.out_bias.fill_(20.0)
        atomic_model.out_std.fill_(3.0)
    torch.testing.assert_close(model(coord, atype)["phase"], before)


def test_classification_input_statistics_with_scalar_class_ids() -> None:
    config = _config()
    config["fitting_net"]["numb_fparam"] = 1
    model = get_model(config).cpu()
    coord, atype = _inputs()
    samples = [
        {
            "coord": coord.repeat(2, 1, 1),
            "atype": atype.repeat(2, 1),
            "phase": torch.tensor([[0], [2]]),
            "find_phase": True,
            "fparam": torch.tensor([[2.0], [4.0]], dtype=torch.float64),
            "find_fparam": True,
        }
    ]
    model.atomic_model.compute_or_load_stat(lambda: samples)
    torch.testing.assert_close(
        model.get_fitting_net().fparam_avg, torch.tensor([3.0], dtype=torch.float64)
    )
    assert torch.count_nonzero(model.atomic_model.out_bias) == 0
    assert torch.all(model.atomic_model.out_std == 1)


def test_classification_portable_roundtrip() -> None:
    model = get_model(_config()).cpu()
    coord, atype = _inputs()
    expected = model(coord, atype)["phase"].detach().numpy()
    serialized = model.serialize()
    assert serialized["fitting"]["type"] == "classification"
    assert serialized["fitting"]["num_classes"] == 3
    restored = BaseModel.deserialize(serialized).cpu()
    np.testing.assert_allclose(restored(coord, atype)["phase"].detach(), expected)
    portable = DPBaseModel.deserialize(serialized)
    np.testing.assert_allclose(
        portable(coord.numpy(), atype.numpy())["phase"], expected, atol=1e-10
    )


def test_classification_ragged_mean_and_padding() -> None:
    model = get_model(_config()).cpu()
    coord, atype = _inputs()
    # Two different frame sizes, compared to independent dense calls.
    flat_coord = torch.cat([coord[0], coord[0, :2]], dim=0)
    flat_atype = torch.cat([atype[0], atype[0, :2]], dim=0)
    n_node = torch.tensor([3, 2])
    ragged = model.forward_ragged(flat_coord, flat_atype, n_node)
    expected = torch.cat(
        [model(coord, atype)["phase"], model(coord[:, :2], atype[:, :2])["phase"]]
    )
    torch.testing.assert_close(ragged["phase"], expected)
    torch.testing.assert_close(ragged["phase"][1], ragged["atom_phase"][3:].mean(dim=0))
    padded_atype = torch.tensor([[0, 1, -1]])
    padded = model(coord, padded_atype)
    torch.testing.assert_close(padded["phase"], expected[1:2])


def test_energy_named_class_label_stays_classification() -> None:
    config = _config()
    config["fitting_net"]["label_name"] = "energy"
    model = get_model(config).cpu()
    assert model.get_is_classification()
    assert not model_uses_graph_lower(model)
    assert model._resolve_graph_method(None) is None
    coord, atype = _inputs()
    assert model(coord, atype)["energy"].shape == (1, 3)


def test_dpa4_classification_export_inference(tmp_path) -> None:
    """A DPA4 export retains geometry-dependent logits and dynamic batches."""
    model = get_model(_config()).cpu().eval()
    # Activate DPA4's initially zero residual projections so this exercises
    # geometry-dependent predictions, rather than only the type embedding.
    serialized = jitter_zero_arrays(model.serialize(), np.random.default_rng(99))
    model = BaseModel.deserialize(serialized).cpu().eval()
    coord, atype = _inputs()
    coord = torch.cat([coord, coord * 1.25])
    atype = atype.repeat(2, 1)
    expected = model(coord, atype)
    assert not torch.allclose(expected["phase"][0], expected["phase"][1])
    export_path = tmp_path / "dpa4_classification.pte"
    deserialize_to_file(str(export_path), {"model": serialized})
    with DeepEval(str(export_path), auto_batch_size=False) as evaluator:
        assert isinstance(evaluator, DeepClassification)
        assert evaluator.get_num_classes() == 3
        for nframes in (1, 2):
            logits, atomic_logits = evaluator.eval(
                coord[:nframes].numpy(), None, atype[0].numpy(), atomic=True
            )
            np.testing.assert_allclose(
                logits, expected["phase"][:nframes].detach().numpy(), atol=1e-10
            )
            np.testing.assert_allclose(
                atomic_logits,
                expected["atom_phase"][:nframes].detach().numpy(),
                atol=1e-10,
            )
