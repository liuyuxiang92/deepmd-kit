# SPDX-License-Identifier: LGPL-3.0-or-later
"""Inference for system-level single-label classification."""

from typing import (
    Any,
)

import numpy as np

from deepmd.infer.deep_property import (
    DeepProperty,
)


class DeepClassification(DeepProperty):
    """Evaluate raw class logits, probabilities, or predicted class indices.

    The inherited :meth:`eval` returns ``(logits,)`` with shape
    ``(nframes, num_classes)``. With ``atomic=True`` it additionally returns
    per-atom logits, whose mean over real atoms gives the system logits.
    """

    def get_num_classes(self) -> int:
        """Return the number of classes."""
        return self.get_task_dim()

    def predict_proba(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Return class probabilities with the same inputs as :meth:`eval`."""
        logits = self.eval(*args, **kwargs)[0]
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum(axis=-1, keepdims=True)

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Return one integer class index per frame, with :meth:`eval` inputs."""
        return np.argmax(self.eval(*args, **kwargs)[0], axis=-1)


__all__ = ["DeepClassification"]
