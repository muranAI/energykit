"""
energykit.forecast.load
========================
Energy load forecaster with a scikit-learn compatible API.

Uses gradient boosting (LightGBM when available, otherwise scikit-learn's
``HistGradientBoostingRegressor``) combined with automatically generated
energy-specific features from :class:`~energykit.features.EnergyFeatureExtractor`.

The forecast strategy is *recursive multi-step*: a single model is trained
to predict one step ahead, then predictions are fed back as lag inputs for
subsequent horizons.  This gives good out-of-the-box accuracy without
requiring a separate model per horizon.

Usage
-----
>>> from energykit.forecast import LoadForecaster
>>> model = LoadForecaster(horizon=24, country="US")
>>> model.fit(meter_series)           # pd.Series with hourly DatetimeIndex
>>> forecast = model.predict()        # pd.Series, next 24 hours
>>> print(forecast)
2026-03-07 00:00:00    452.3
2026-03-07 01:00:00    431.7
...

Custom horizon at predict time
>>> forecast_48h = model.predict(horizon=48)
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from energykit.features.temporal import EnergyFeatureExtractor

# Prefer LightGBM if available — faster and slightly more accurate on tabular data
try:
    import lightgbm as lgb  # type: ignore

    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


def _build_model(model_params: Optional[Dict] = None):
    """Return a gradient boosting regressor with sensible energy defaults."""
    if _HAS_LGB:
        default_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "verbose": -1,
        }
        if model_params:
            default_params.update(model_params)
        return lgb.LGBMRegressor(**default_params)

    # Fallback: sklearn HistGradientBoosting (no extra deps, handles NaN natively)
    default_params = {
        "max_iter": 500,
        "learning_rate": 0.05,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
    }
    if model_params:
        default_params.update(model_params)
    return HistGradientBoostingRegressor(**default_params)


class LoadForecaster(BaseEstimator):
    """Energy load forecaster powered by gradient boosting + energy features.

    Parameters
    ----------
    horizon : int, default 24
        Default forecast horizon in periods (hours for hourly data).
    lags : list of int or None
        Lag features to include. Defaults to ``[1, 2, 3, 24, 48, 168]``.
    rolling_windows : list of int or None
        Rolling statistic windows. Defaults to ``[24, 168]``.
    country : str or None
        ISO country code for holiday features (e.g. ``"US"``, ``"DE"``).
    lat, lon : float or None
        Coordinates for solar position features.
    model_params : dict or None
        Override default LightGBM / HGBR hyperparameters.
    tou_schedule : dict or None
        Custom time-of-use schedule passed to the feature extractor.

    Attributes
    ----------
    model_ : fitted regressor
        The underlying gradient boosting model.
    feature_extractor_ : EnergyFeatureExtractor
        Fitted feature extractor (stateless, but retains ``feature_names_``).
    train_series_ : pd.Series
        The training series (kept for recursive prediction warm-start).
    is_fitted_ : bool
        ``True`` after :meth:`fit` has been called.

    Examples
    --------
    >>> model = LoadForecaster(horizon=24, country="US")
    >>> model.fit(meter)
    >>> fc = model.predict()
    """

    def __init__(
        self,
        horizon: int = 24,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        country: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        model_params: Optional[Dict] = None,
        tou_schedule=None,
    ) -> None:
        self.horizon = horizon
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.country = country
        self.lat = lat
        self.lon = lon
        self.model_params = model_params
        self.tou_schedule = tou_schedule

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: pd.Series, y=None) -> "LoadForecaster":
        """Train the forecaster on historical load data.

        Parameters
        ----------
        X : pd.Series
            Hourly smart-meter or aggregated load readings with a
            ``DatetimeIndex``. Gaps and ``NaN`` values are forward-filled
            before training.

        Returns
        -------
        self
        """
        if not isinstance(X, pd.Series):
            raise TypeError("X must be a pd.Series with a DatetimeIndex.")
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X.index must be a DatetimeIndex.")
        if len(X) < max(self.lags or [168]) + 2:
            raise ValueError(
                f"Training series too short. Need at least "
                f"{max(self.lags or [168]) + 2} observations."
            )

        # Fill gaps
        series = X.copy().ffill().bfill()
        self.train_series_ = series

        # Build feature extractor
        self.feature_extractor_ = EnergyFeatureExtractor(
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            country=self.country,
            lat=self.lat,
            lon=self.lon,
            tou_schedule=self.tou_schedule,
        )

        # Generate features and align target
        feats = self.feature_extractor_.fit_transform(series)
        target = series.values

        # Drop rows with NaN (from lags at start of series)
        valid_mask = ~np.isnan(feats.values).any(axis=1)
        feats_clean = feats.values[valid_mask]
        target_clean = target[valid_mask]

        # Fit model
        self.model_ = _build_model(self.model_params)
        self.model_.fit(feats_clean, target_clean)
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        horizon: Optional[int] = None,
        last_known: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Generate a multi-step ahead load forecast.

        Parameters
        ----------
        horizon : int or None
            Forecast horizon in periods. Defaults to ``self.horizon``.
        last_known : pd.Series or None
            Provide a different warm-start series (e.g. latest observations
            from production). Defaults to the training series.

        Returns
        -------
        pd.Series
            Forecasted values with a ``DatetimeIndex`` starting at
            ``last_known.index[-1] + freq``.
        """
        self._check_is_fitted()
        h = horizon if horizon is not None else self.horizon
        series = (last_known if last_known is not None else self.train_series_).copy()

        # Infer data frequency
        freq = self._infer_freq(series)
        forecast_values: List[float] = []

        for _ in range(h):
            feats = self.feature_extractor_.transform(series)
            # Use last row for prediction
            last_feat = feats.iloc[[-1]].values
            # Replace any remaining NaN with 0 (should not happen after warm-start)
            last_feat = np.nan_to_num(last_feat, nan=0.0)
            pred = float(self.model_.predict(last_feat)[0])
            forecast_values.append(pred)

            # Append prediction to series for next step
            next_ts = series.index[-1] + pd.tseries.frequencies.to_offset(freq)
            series = pd.concat(
                [series, pd.Series([pred], index=[next_ts], name=series.name)]
            )

        forecast_index = pd.date_range(
            start=self.train_series_.index[-1] + pd.tseries.frequencies.to_offset(freq)
            if last_known is None
            else last_known.index[-1] + pd.tseries.frequencies.to_offset(freq),
            periods=h,
            freq=freq,
        )
        return pd.Series(forecast_values, index=forecast_index, name="forecast_kwh")

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.Series:
        """Return feature importances sorted descending.

        Works with LightGBM and sklearn HGBR.

        Returns
        -------
        pd.Series
            Feature name → importance score.
        """
        self._check_is_fitted()
        names = self.feature_extractor_.feature_names_
        if _HAS_LGB and hasattr(self.model_, "feature_importances_"):
            importances = self.model_.feature_importances_
        elif hasattr(self.model_, "feature_importances_"):
            importances = self.model_.feature_importances_
        else:
            warnings.warn("Model does not expose feature_importances_.")
            return pd.Series(dtype=float)
        return (
            pd.Series(importances, index=names)
            .sort_values(ascending=False)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit() before predict().")

    @staticmethod
    def _infer_freq(series: pd.Series) -> str:
        """Infer the dominant frequency of a time series."""
        if series.index.freq is not None:
            return series.index.freq.freqstr
        if len(series) >= 3:
            diffs = series.index[1:] - series.index[:-1]
            dominant = pd.Series(diffs).mode().iloc[0]
            # Round to nearest standard freq
            minutes = dominant.total_seconds() / 60
            if abs(minutes - 60) < 5:
                return "h"
            if abs(minutes - 30) < 5:
                return "30min"
            if abs(minutes - 15) < 5:
                return "15min"
            if abs(minutes - 1440) < 60:
                return "D"
        return "h"
