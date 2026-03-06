"""
energykit.datasets.loaders
===========================
Loaders for public energy datasets and synthetic data generators.

Functions
---------
load_uci_household
    UCI Machine Learning Repository — Individual Household Electric Power
    Consumption. Minute-level active power for a French household (2006–2010).
    Automatically downloaded and cached on first use.

load_synthetic_load
    Generate realistic synthetic hourly load profiles with configurable
    seasonality, peak patterns, and noise. Useful for testing and
    demo notebooks without requiring real data.

load_sample_tou_prices
    Return a sample 24-hour Time-of-Use price vector representative of
    common US utility tariffs ($/kWh, hourly resolution).
"""

from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Default cache directory — respects XDG_CACHE_HOME on Linux/macOS
_CACHE_DIR = Path(os.getenv("ENERGYKIT_CACHE", Path.home() / ".cache" / "energykit"))

# UCI dataset constants
_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00235/household_power_consumption.zip"
)
_UCI_FILENAME = "household_power_consumption.txt"
_UCI_SHA256 = "9dc57a6eddde55ef8c7b1c1d85b6e16b40348a7d0b5b1b78d7b1b"  # partial


def load_uci_household(
    resample: str = "h",
    cache_dir: Optional[Path] = None,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """Load the UCI Household Power Consumption dataset.

    Downloads and caches the dataset on first call (~20 MB zip).
    Subsequent calls load from the local cache.

    Parameters
    ----------
    resample : str, default ``"h"``
        Resample frequency. ``"1min"`` returns raw minute-level data.
        Common options: ``"h"``, ``"15min"``, ``"30min"``, ``"D"``.
    cache_dir : Path or None
        Override the default cache directory
        (``~/.cache/energykit`` or ``$ENERGYKIT_CACHE``).
    columns : list or None
        Subset of columns to return. Available:
        ``Global_active_power``, ``Global_reactive_power``,
        ``Voltage``, ``Global_intensity``,
        ``Sub_metering_1``, ``Sub_metering_2``, ``Sub_metering_3``.

    Returns
    -------
    pd.DataFrame
        DataFrame with a ``DatetimeIndex`` resampled as requested.

    Examples
    --------
    >>> df = load_uci_household(resample="h")
    >>> active_power = df["Global_active_power"]  # kW
    """
    cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    txt_path = cache_dir / _UCI_FILENAME

    if not txt_path.exists():
        _download_uci(cache_dir)

    df = _parse_uci(txt_path)

    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}. Available: {list(df.columns)}")
        df = df[columns]

    if resample != "1min":
        df = df.resample(resample).mean()

    return df


def _download_uci(cache_dir: Path) -> None:
    """Download and extract the UCI dataset zip."""
    try:
        import requests  # type: ignore
        from tqdm import tqdm  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Install download dependencies: pip install energykit[datasets]"
        ) from e

    zip_path = cache_dir / "household_power_consumption.zip"
    print(f"Downloading UCI Household Power Consumption dataset → {zip_path}")

    with requests.get(_UCI_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="UCI dataset"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(_UCI_FILENAME, cache_dir)

    zip_path.unlink()  # Remove zip after extraction


def _parse_uci(path: Path) -> pd.DataFrame:
    """Parse the raw UCI text file into a clean DataFrame."""
    df = pd.read_csv(
        path,
        sep=";",
        na_values="?",
        infer_datetime_format=True,
        parse_dates={"datetime": ["Date", "Time"]},
        index_col="datetime",
        dayfirst=True,
        low_memory=False,
    )
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill()
    # Global_active_power is in kW already in the UCI dataset
    return df


# ---------------------------------------------------------------------------
# Synthetic load generator
# ---------------------------------------------------------------------------

def load_synthetic_load(
    start: str = "2025-01-01",
    periods: int = 8760,
    freq: str = "h",
    base_kw: float = 2.5,
    peak_kw: float = 5.0,
    noise_std: float = 0.3,
    seed: int = 42,
    country: Optional[str] = None,
) -> pd.Series:
    """Generate a synthetic hourly residential load profile.

    The profile combines:
    - Annual seasonality (winter peak / summer AC peak)
    - Daily profile (morning ramp, evening peak)
    - Weekday/weekend differentiation
    - Random Gaussian noise

    Parameters
    ----------
    start : str
        Start date of the series (ISO format).
    periods : int, default 8760
        Number of periods (8760 = 1 year of hourly data).
    freq : str, default ``"h"``
        Data frequency.
    base_kw : float
        Minimum (overnight) load in kW.
    peak_kw : float
        Maximum (evening peak) load in kW.
    noise_std : float
        Standard deviation of Gaussian noise (kW).
    seed : int
        Random seed for reproducibility.
    country : str or None
        Not currently used. Reserved for future holiday dip modelling.

    Returns
    -------
    pd.Series
        Synthetic load in kW with a ``DatetimeIndex``.

    Examples
    --------
    >>> load = load_synthetic_load(periods=8760)
    >>> load.plot(title="Synthetic Load Profile")
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=periods, freq=freq)

    hour = idx.hour.values.astype(float)
    dow = idx.dayofweek.values.astype(float)
    doy = idx.dayofyear.values.astype(float)

    # Daily profile: morning ramp + evening peak
    daily = (
        0.3 * np.exp(-0.5 * ((hour - 7) / 2) ** 2)   # morning ramp
        + 0.7 * np.exp(-0.5 * ((hour - 19) / 2.5) ** 2)  # evening peak
        + 0.1 * np.exp(-0.5 * ((hour - 12) / 1.5) ** 2)  # midday dip
    )
    daily /= daily.max()

    # Annual seasonality: winter heating + summer AC
    annual = 0.4 * np.cos(2 * np.pi * (doy - 15) / 365) + 0.6

    # Weekend reduction
    weekend_factor = np.where(dow >= 5, 0.85, 1.0)

    load = base_kw + (peak_kw - base_kw) * daily * annual * weekend_factor
    load += rng.normal(0, noise_std, size=periods)
    load = np.maximum(load, base_kw * 0.3)  # floor

    return pd.Series(load, index=idx, name="load_kw")


# ---------------------------------------------------------------------------
# Sample TOU prices
# ---------------------------------------------------------------------------

def load_sample_tou_prices(
    tariff: str = "residential_us",
    periods: int = 24,
) -> np.ndarray:
    """Return sample Time-of-Use electricity prices ($/kWh).

    Parameters
    ----------
    tariff : str
        Price profile to return:
        - ``"residential_us"`` — US residential TOU (3-block: off/mid/on peak)
        - ``"flat_us"`` — US average flat rate (~$0.16/kWh)
        - ``"wholesale_eu"`` — Simulated day-ahead European wholesale prices
    periods : int, default 24
        Number of hourly periods to return (repeats if > 24).

    Returns
    -------
    np.ndarray, shape (periods,)
        Electricity prices in $/kWh.
    """
    profiles = {
        "residential_us": np.array(
            [0.09] * 7       # 00:00–06:59 off-peak
            + [0.14] * 10    # 07:00–16:59 mid-peak
            + [0.28] * 5     # 17:00–21:59 on-peak
            + [0.09] * 2,    # 22:00–23:59 off-peak
        ),
        "flat_us": np.full(24, 0.16),
        "wholesale_eu": np.array(
            [0.04, 0.03, 0.03, 0.03, 0.04, 0.05,
             0.08, 0.12, 0.15, 0.14, 0.13, 0.12,
             0.11, 0.12, 0.13, 0.14, 0.18, 0.22,
             0.25, 0.20, 0.16, 0.12, 0.08, 0.05]
        ),
    }
    if tariff not in profiles:
        raise ValueError(f"Unknown tariff {tariff!r}. Choose from {list(profiles)}")

    base = profiles[tariff]
    if periods <= 24:
        return base[:periods]
    reps = (periods // 24) + 1
    return np.tile(base, reps)[:periods]
