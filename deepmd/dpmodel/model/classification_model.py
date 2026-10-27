# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.atomic_model import (
    DPClassificationAtomicModel,
)
from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.output_def import (
    OutputVariableDef,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)

DPClassificationModel_ = make_model(
    DPClassificationAtomicModel, T_Bases=(NativeOP, BaseModel)
)


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

    def get_task_dim(self) -> int:
        """Get the number of classes."""
        return self.get_fitting_net().dim_out

    def get_intensive(self) -> bool:
        """Class logits are averaged over real atoms."""
        return self.model_output_def()[self.get_var_name()].intensive

    def call(
        self,
        coord: Array,
        atype: Array,
        box: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        do_atomic_virial: bool = False,
        charge_spin: Array | None = None,
    ) -> dict[str, Array]:
        model_ret = self.call_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
        )
        var_name = self.get_var_name()
        model_predict = {}
        model_predict[f"atom_{var_name}"] = model_ret[var_name]
        model_predict[var_name] = model_ret[f"{var_name}_redu"]
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    def call_lower(
        self,
        extended_coord: Array,
        extended_atype: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        do_atomic_virial: bool = False,
        charge_spin: Array | None = None,
    ) -> dict[str, Array]:
        model_ret = self.call_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
        )
        var_name = self.get_var_name()
        model_predict = {}
        model_predict[f"atom_{var_name}"] = model_ret[var_name]
        model_predict[var_name] = model_ret[f"{var_name}_redu"]
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    def translated_output_def(self) -> dict[str, OutputVariableDef]:
        out_def_data = self.model_output_def().get_data()
        var_name = self.get_var_name()
        output_def = {
            f"atom_{var_name}": out_def_data[var_name],
            var_name: out_def_data[f"{var_name}_redu"],
        }
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        return output_def
