# SPDX-License-Identifier: LGPL-3.0-or-later
import types
from typing import (
    Any,
)

import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.atomic_model import (
    DPClassificationAtomicModel,
)
from deepmd.dpmodel.model.dp_model import (
    DPModelCommon,
)

from .make_model import (
    _pad_nlist_for_export,
    make_model,
)
from .model import (
    BaseModel,
)

DPClassificationModel_ = make_model(DPClassificationAtomicModel, T_Bases=(BaseModel,))


@BaseModel.register("classification")
class ClassificationModel(DPModelCommon, DPClassificationModel_):
    """System-level class logits, averaged over real atoms."""

    model_type = "classification"

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        DPClassificationModel_.__init__(self, *args, **kwargs)

    def get_var_name(self) -> str:
        """Get the name of the class label."""
        return self.get_fitting_net().var_name

    def get_num_classes(self) -> int:
        """Get the number of classes."""
        return self.get_fitting_net().num_classes

    def get_is_classification(self) -> bool:
        """Identify categorical outputs for inference and evaluation."""
        return True

    def _resolve_graph_method(self, neighbor_graph_method: str | None) -> str | None:
        """Use the property path even when a class label is named ``energy``."""
        if neighbor_graph_method == "legacy":
            return None
        return neighbor_graph_method

    def get_task_dim(self) -> int:
        """Get the number of classes."""
        return self.get_fitting_net().dim_out

    def get_intensive(self) -> bool:
        """Class logits are averaged over real atoms."""
        return self.model_output_def()[self.get_var_name()].intensive

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.call_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
            do_atomic_virial=do_atomic_virial,
        )
        var_name = self.get_var_name()
        model_predict = {}
        model_predict[f"atom_{var_name}"] = model_ret[var_name]
        model_predict[var_name] = model_ret[f"{var_name}_redu"]
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.call_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
            do_atomic_virial=do_atomic_virial,
        )
        var_name = self.get_var_name()
        model_predict = {}
        model_predict[f"atom_{var_name}"] = model_ret[var_name]
        model_predict[var_name] = model_ret[f"{var_name}_redu"]
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    def forward_ragged(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        n_node: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate frames with different atom counts on a flat node axis."""
        result = self.call_common_ragged(
            coord,
            atype,
            n_node,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
        )
        var_name = self.get_var_name()
        output = {
            f"atom_{var_name}": result[var_name],
            var_name: result[f"{var_name}_redu"],
            "n_node": n_node,
        }
        if "mask" in result:
            output["mask"] = result["mask"]
        return output

    def translated_output_def(self) -> dict[str, Any]:
        out_def_data = self.model_output_def().get_data()
        var_name = self.get_var_name()
        output_def = {
            f"atom_{var_name}": out_def_data[var_name],
            var_name: out_def_data[f"{var_name}_redu"],
        }
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        return output_def

    def forward_lower_exportable(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
        **make_fx_kwargs: Any,
    ) -> torch.nn.Module:
        model = self

        def fn(
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            nlist: torch.Tensor,
            mapping: torch.Tensor | None,
            fparam: torch.Tensor | None,
            aparam: torch.Tensor | None,
            charge_spin: torch.Tensor | None,
        ) -> dict[str, torch.Tensor]:
            extended_coord = extended_coord.detach().requires_grad_(True)
            nlist = _pad_nlist_for_export(nlist)
            return model.forward_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fparam=fparam,
                aparam=aparam,
                charge_spin=charge_spin,
                do_atomic_virial=do_atomic_virial,
            )

        # See make_model.py for the rationale of the pad + monkeypatch.
        _orig_need_sort = model.need_sorted_nlist_for_lower
        model.need_sorted_nlist_for_lower = types.MethodType(lambda self: True, model)
        try:
            traced = make_fx(fn, **make_fx_kwargs)(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fparam,
                aparam,
                charge_spin,
            )
        finally:
            model.need_sorted_nlist_for_lower = _orig_need_sort
        return traced
