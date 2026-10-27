# SPDX-License-Identifier: LGPL-3.0-or-later
"""Classification accuracy and cross-entropy for ``dp test``."""

import numpy as np

from deepmd.dpmodel.loss.classification import (
    ClassificationLoss,
)
from deepmd.infer.model_test.base import (
    ChunkContext,
    ModelTester,
    _write_per_frame_details,
)
from deepmd.utils.data import (
    DeepmdData,
)


class ClassificationTester(ModelTester):
    """Test one class label per frame, with frame-weighted aggregate metrics."""

    report = (
        ("accuracy", "Classification accuracy : {}"),
        ("cross_entropy", "Cross entropy           : {}"),
    )

    def add_data_requirements(self, data: DeepmdData) -> None:
        """Require scalar labels and any fitting parameters."""
        dp = self.dp
        data.add(dp.get_var_name(), 1, atomic=False, must=True, high_prec=True)
        if dp.get_dim_fparam() > 0:
            data.add(
                "fparam", dp.get_dim_fparam(), atomic=False, must=True, high_prec=False
            )
        if dp.get_dim_aparam() > 0:
            data.add(
                "aparam", dp.get_dim_aparam(), atomic=True, must=True, high_prec=False
            )

    def evaluate_chunk(
        self,
        data: DeepmdData,
        test_data: dict,
        context: ChunkContext,
    ) -> dict[str, tuple[float, float]]:
        """Evaluate raw logits and validate the labels before computing metrics."""
        dp = self.dp
        var_name = dp.get_var_name()
        nframes = test_data["box"].shape[0]
        atype = (
            test_data["type"].reshape(nframes, -1)
            if data.mixed_type
            else test_data["type"][0]
        )
        logits = dp.eval(
            test_data["coord"].reshape(nframes, -1),
            test_data["box"] if data.pbc else None,
            atype,
            atomic=self.atomic,
            fparam=test_data["fparam"] if dp.get_dim_fparam() > 0 else None,
            aparam=test_data["aparam"] if dp.get_dim_aparam() > 0 else None,
            mixed_type=data.mixed_type,
        )[0]
        loss, metrics = ClassificationLoss(dp.get_num_classes(), var_name)(
            0.0,
            len(test_data["type"][0]),
            {var_name: logits},
            {var_name: test_data[var_name]},
        )
        if context.detail_path is not None:
            _write_per_frame_details(
                context,
                suffix="classification",
                reference=test_data[var_name],
                prediction=np.argmax(logits, axis=-1),
            )
        return {
            "accuracy": (float(metrics["accuracy"]), nframes),
            "cross_entropy": (float(loss), nframes),
        }
