"""ARMA feature extraction helpers."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # statsmodels is an optional dependency for ARMA fitting
    from statsmodels.tsa.arima.model import ARIMA
except ImportError as exc:  # pragma: no cover - module level guard
    raise ImportError(
        "statsmodels is required for ARMA feature extraction. "
        "Please install it via `pip install statsmodels`."
    ) from exc

ArrayLike = Union[np.ndarray, Sequence[float]]


def _ensure_2d(signal: ArrayLike) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim == 1:
        return arr[None, :], True
    if arr.ndim == 2:
        return arr, False
    raise ValueError(f"Expected a 1-D or 2-D array, got shape {arr.shape} instead.")


def _default_result(length: int) -> Dict[str, Union[int, float, List[float]]]:
    return {
        "order": (0, 0),
        "ar_params": [],
        "ma_params": [],
        "sigma2": 0.0,
        "aic": float("nan"),
        "bic": float("nan"),
        "log_likelihood": float("nan"),
        "nobs": length,
    }


def fit_arma(
    signal: ArrayLike,
    *,
    max_order: Tuple[int, int] = (4, 4),
    information_criterion: str = "aic",
    allow_constant: bool = False,
) -> Union[Dict[str, Union[int, float, List[float]]], List[Dict[str, Union[int, float, List[float]]]]]:
    """Fit an ARMA(p, q) model and return compact descriptors.

    Parameters
    ----------
    signal:
        Input vibration sequence(s). Supports 1-D or 2-D arrays.
    max_order:
        Upper bound for the AR (p) and MA (q) orders explored during grid search.
    information_criterion:
        Either ``"aic"`` or ``"bic"``; used to select the best model during search.
    allow_constant:
        Whether to include a constant term in the ARMA model.
    """

    if information_criterion not in {"aic", "bic"}:
        raise ValueError("information_criterion must be either 'aic' or 'bic'.")

    arr, squeeze = _ensure_2d(signal)
    results: List[Dict[str, Union[int, float, List[float]]]] = []

    for row in arr:
        best_result = None
        best_ic = float("inf")
        best_order: Optional[Tuple[int, int]] = None

        for p in range(max_order[0] + 1):
            for q in range(max_order[1] + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(row, order=(p, 0, q), trend="c" if allow_constant else "n")
                    fitted = model.fit(method_kwargs={"warn_convergence": False})
                    ic_value = getattr(fitted, information_criterion)
                except Exception:
                    continue
                if np.isnan(ic_value):
                    continue
                if ic_value < best_ic:
                    best_ic = ic_value
                    best_result = fitted
                    best_order = (p, q)

        if best_result is None or best_order is None:
            results.append(_default_result(length=row.size))
            continue

        params_obj = best_result.params
        if hasattr(params_obj, "to_dict"):
            params_dict = params_obj.to_dict()
        else:
            param_names = getattr(best_result, "param_names", [])
            params_array = np.asarray(params_obj)
            params_dict = {
                name: float(value)
                for name, value in zip(param_names, params_array.tolist())
            }

        ar_params = [float(params_dict.get(f"ar.L{i+1}", 0.0)) for i in range(best_order[0])]
        ma_params = [float(params_dict.get(f"ma.L{i+1}", 0.0)) for i in range(best_order[1])]
        summary = {
            "order": best_order,
            "ar_params": ar_params,
            "ma_params": ma_params,
            "sigma2": float(getattr(best_result, "sigma2", 0.0)),
            "aic": float(best_result.aic),
            "bic": float(best_result.bic),
            "log_likelihood": float(best_result.llf),
            "nobs": int(best_result.nobs),
        }
        results.append(summary)

    return results[0] if squeeze else results


__all__ = ["fit_arma"]
