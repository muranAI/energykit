"""
energykit.features.temporal
============================
Energy-specific feature engineering from time-indexed smart-meter data.

Features generated
------------------
Temporal
    hour, day_of_week, month, week_of_year, day_of_year
    is_weekend, is_holiday (optional, requires ``holidays`` package)
    season (0=winter, 1=spring, 2=summer, 3=autumn)

Cyclical encoding
    sin/cos encoding for hour, day-of-week, month, day-of-year.
    Avoids discontinuities at period boundaries (e.g. hour 23 → hour 0).

Time-of-use blocks
    Configurable on-peak / mid-peak / off-peak binary flags.
    Default schedule follows a typical US utility TOU tariff.

Lag features
    Autoregressive lags at configurable horizons (default: 1h, 2h, 3h, 24h, 48h, 168h).

Rolling statistics
    Rolling mean, std, min, max over configurable windows (default: 24h, 168h).
    Window is applied to *lagged* data to prevent data leakage.

Solar position
    Simplified solar elevation angle (degrees). Requires lat/lon.
    Useful proxy for solar irradiance and PV generation features.

Usage
-----
>>> import pandas as pd
>>> from energykit.features import EnergyFeatureExtractor
>>>
>>> meter = pd.read_csv("meter.csv", index_col=0, parse_dates=True)["kwh"]
>>> fe = EnergyFeatureExtractor(country="US", lat=37.7, lon=-122.4)
>>> features = fe.fit_transform(meter)
>>> print(features.shape)   # (n_samples, n_features)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Optional holiday support
try:
    import holidays as _holidays_lib  # type: ignore

    _HAS_HOLIDAYS = True
except ImportError:  # pragma: no cover
    _HAS_HOLIDAYS = False

# Default TOU schedule (hours in HE notation, typical US utility)
_DEFAULT_TOU: Dict[str, List[int]] = {
    "off_peak": list(range(0, 7)) + list(range(22, 24)),
    "mid_peak": list(range(7, 17)),
    "on_peak": list(range(17, 22)),
}


class EnergyFeatureExtractor(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for energy time-series features.

    Transforms a ``pd.Series`` or ``pd.DataFrame`` with a ``DatetimeIndex``
    into a rich feature matrix suited for load forecasting, anomaly detection,
    and device disaggregation tasks.

    Parameters
    ----------
    lags : list of int, default [1, 2, 3, 24, 48, 168]
        Lag periods (assumes hourly data). Lag features are built from the
        *first column* (or the Series itself).
    rolling_windows : list of int, default [24, 168]
        Rolling window sizes (number of periods). Rolling statistics are
        computed on lag-1 shifted data to prevent data leakage.
    cyclical : bool, default True
        Encode periodic features as sin/cos pairs to avoid boundary
        discontinuities.
    tou_schedule : dict or None
        Time-of-use block mapping ``{block_name: [hour, ...]}`` where hours
        are 0-based integers (0 = midnight). Pass ``{}`` to disable.
        Defaults to a typical US on_peak / mid_peak / off_peak schedule.
    country : str or None
        ISO 3166-1 alpha-2 country code for holiday detection (e.g. ``"US"``,
        ``"DE"``). Requires the ``holidays`` package
        (``pip install holidays``).
    lat : float or None
        Latitude in decimal degrees. Required for solar position features.
    lon : float or None
        Longitude in decimal degrees. Required for solar position features.

    Attributes
    ----------
    feature_names_ : list of str
        Names of all generated features (set after ``fit``/``fit_transform``).

    Examples
    --------
    >>> fe = EnergyFeatureExtractor(country="US", lags=[1, 24, 168])
    >>> X = fe.fit_transform(meter_series)
    >>> X.shape
    (8760, 42)
    """

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        cyclical: bool = True,
        tou_schedule: Optional[Dict[str, List[int]]] = _DEFAULT_TOU,
        country: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> None:
        self.lags = lags if lags is not None else [1, 2, 3, 24, 48, 168]
        self.rolling_windows = rolling_windows if rolling_windows is not None else [24, 168]
        self.cyclical = cyclical
        self.tou_schedule = tou_schedule if tou_schedule is not None else {}
        self.country = country
        self.lat = lat
        self.lon = lon

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y=None):  # noqa: N803
        """No fitting required; features are fully deterministic."""
        return self

    def transform(self, X) -> pd.DataFrame:  # noqa: N803
        """Generate energy features.

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Input data with a ``DatetimeIndex``. If a ``DataFrame``, the
            first column is used for lag and rolling features.

        Returns
        -------
        pd.DataFrame
            Feature matrix indexed identically to *X*.

        Raises
        ------
        ValueError
            If *X* does not have a ``DatetimeIndex``.
        """
        series, idx = self._extract_series_and_index(X)
        features = pd.DataFrame(index=idx)

        self._add_temporal(features, idx)
        self._add_holiday(features, idx)
        if self.cyclical:
            self._add_cyclical(features, idx)
        if self.tou_schedule:
            self._add_tou(features, idx)
        self._add_lags(features, series)
        self._add_rolling(features, series)
        if self.lat is not None and self.lon is not None:
            features["solar_elevation_deg"] = self._solar_elevation(idx)

        self.feature_names_ = list(features.columns)
        return features

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return feature names (sklearn Pipeline compatibility)."""
        return np.array(self.feature_names_)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_series_and_index(X) -> tuple:
        if isinstance(X, pd.Series):
            series = X.copy()
            idx = X.index
        elif isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0].copy()
            idx = X.index
        else:
            raise TypeError(f"Expected pd.Series or pd.DataFrame, got {type(X)}")
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError(
                "Input must have a DatetimeIndex. "
                "Use pd.to_datetime() and set as index first."
            )
        return series, idx

    @staticmethod
    def _add_temporal(df: pd.DataFrame, idx: pd.DatetimeIndex) -> None:
        df["hour"] = idx.hour
        df["day_of_week"] = idx.dayofweek
        df["month"] = idx.month
        df["quarter"] = idx.quarter
        df["week_of_year"] = idx.isocalendar().week.astype(int)
        df["day_of_year"] = idx.dayofyear
        df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
        df["season"] = pd.array(
            [EnergyFeatureExtractor._month_to_season(m) for m in idx.month],
            dtype="int8",
        )

    def _add_holiday(self, df: pd.DataFrame, idx: pd.DatetimeIndex) -> None:
        if self.country and _HAS_HOLIDAYS:
            try:
                cal = _holidays_lib.country_holidays(self.country)
                holiday_dates = pd.DatetimeIndex(list(cal.keys()))
                df["is_holiday"] = idx.normalize().isin(holiday_dates).astype(int)
                return
            except Exception:  # noqa: BLE001
                pass
        df["is_holiday"] = 0

    @staticmethod
    def _add_cyclical(df: pd.DataFrame, idx: pd.DatetimeIndex) -> None:
        df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
        df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
        df["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365)
        df["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365)

    def _add_tou(self, df: pd.DataFrame, idx: pd.DatetimeIndex) -> None:
        for block, hours in self.tou_schedule.items():
            df[f"tou_{block}"] = idx.hour.isin(hours).astype(int)

    def _add_lags(self, df: pd.DataFrame, series: pd.Series) -> None:
        for lag in self.lags:
            df[f"lag_{lag}h"] = series.shift(lag).values

    def _add_rolling(self, df: pd.DataFrame, series: pd.Series) -> None:
        # Shift by 1 to avoid data leakage — rolling window ends *before*
        # the current observation.
        lagged = series.shift(1)
        for window in self.rolling_windows:
            rolled = lagged.rolling(window=window, min_periods=1)
            df[f"roll_{window}h_mean"] = rolled.mean().values
            df[f"roll_{window}h_std"] = rolled.std().fillna(0).values
            df[f"roll_{window}h_min"] = rolled.min().values
            df[f"roll_{window}h_max"] = rolled.max().values

    def _solar_elevation(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """Simplified NOAA solar elevation angle (±1° accuracy).

        Returns elevation in degrees above horizon. Negative values mean
        sun is below the horizon.
        """
        lat_rad = np.radians(self.lat)
        n = idx.dayofyear.values.astype(float)

        # Solar declination
        declination = np.radians(23.45 * np.sin(np.radians(360.0 * (n - 81) / 365.0)))

        # Equation of time (minutes)
        B = np.radians(360.0 * (n - 81) / 365.0)
        eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)

        # Solar time correction (assume UTC offset from tzinfo when available)
        tz_hours = self._tz_offset_hours(idx)
        time_correction_h = (eot + 4.0 * self.lon) / 60.0 + tz_hours
        solar_time = idx.hour.values + idx.minute.values / 60.0 + time_correction_h

        # Hour angle
        hour_angle = np.radians(15.0 * (solar_time - 12.0))

        elevation = np.degrees(
            np.arcsin(
                np.sin(lat_rad) * np.sin(declination)
                + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)
            )
        )
        return elevation

    @staticmethod
    def _tz_offset_hours(idx: pd.DatetimeIndex) -> float:
        """Return the UTC offset in hours (scalar), or 0.0 if timezone-naive."""
        if idx.tzinfo is not None:
            try:
                offset = idx[0].utcoffset()
                if offset is not None:
                    return offset.total_seconds() / 3600.0
            except Exception:  # noqa: BLE001
                pass
        return 0.0

    @staticmethod
    def _month_to_season(month: int) -> int:
        """Return meteorological season index (0=winter, 1=spring, 2=summer, 3=autumn)."""
        if month in (12, 1, 2):
            return 0
        if month in (3, 4, 5):
            return 1
        if month in (6, 7, 8):
            return 2
        return 3
