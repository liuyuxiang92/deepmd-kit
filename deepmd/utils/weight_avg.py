# SPDX-License-Identifier: LGPL-3.0-or-later
from collections import (
    defaultdict,
)

import numpy as np


def merge_weighted_errors(
    errors: list[dict[str, tuple[float, float]]],
) -> dict[str, tuple[float, float]]:
    """Combine prediction errors, keeping the weight they were combined over.

    Accuracy and cross-entropy are ordinary frame means. An MAE is the mean
    of the absolute errors and an RMSE the root of the mean
    of their squares, so both are recovered exactly from the partial results by
    weighting the mean, respectively the squared mean, by the number of
    elements each was taken over. Combining partial results is therefore
    equivalent to evaluating the whole set at once, which lets a caller
    evaluate in chunks.

    Parameters
    ----------
    errors : list[dict[str, tuple[float, float]]]
        One ``{quantity: (error, weight)}`` mapping per partial result. A
        quantity name starts with ``mae`` or ``rmse``, or is ``accuracy``
        or ``cross_entropy``.

    Returns
    -------
    dict[str, tuple[float, float]]
        The combined ``(error, weight)`` of every quantity, itself suitable as
        one partial result of a further combination.

    Raises
    ------
    RuntimeError
        If a quantity name does not identify a supported metric.
    """
    sum_err: dict[str, float] = defaultdict(float)
    sum_siz: dict[str, float] = defaultdict(float)
    for err in errors:
        for kk, (ee, ss) in err.items():
            if kk.startswith("mae") or kk in {"accuracy", "cross_entropy"}:
                sum_err[kk] += ee * ss
            elif kk.startswith("rmse"):
                sum_err[kk] += ee * ee * ss
            else:
                raise RuntimeError("unknown error type")
            sum_siz[kk] += ss
    merged: dict[str, tuple[float, float]] = {}
    for kk, total in sum_err.items():
        weight = sum_siz[kk]
        mean = total / weight
        merged[kk] = (float(np.sqrt(mean)) if kk.startswith("rmse") else mean, weight)
    return merged


def weighted_average(errors: list[dict[str, tuple[float, float]]]) -> dict:
    """Compute weighted MAE, RMSE, accuracy or cross-entropy for a model.

    Parameters
    ----------
    errors : list[dict[str, tuple[float, float]]]
        List: the error of systems
        Dict: the error of quantities, name given by the key
        str: a name starting with 'mae' or 'rmse', or 'accuracy'/'cross_entropy'
        Tuple: (error, weight)

    Returns
    -------
    Dict
        weighted averages
    """
    return {kk: value for kk, (value, _) in merge_weighted_errors(errors).items()}
