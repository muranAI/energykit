# Contributing to energykit

Thank you for your interest in contributing! energykit is an open-source project and contributions are welcome from anyone — whether you're fixing a typo, adding a dataset loader, or implementing a new neural forecaster.

## Getting started

```bash
# Fork and clone the repo
git clone https://github.com/<your-username>/energykit.git
cd energykit

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate         # Linux/macOS
.venv\Scripts\activate            # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests to confirm setup
pytest tests/ -v
```

## Development workflow

1. Create a branch: `git checkout -b feat/my-feature`
2. Make your changes
3. Add or update tests in `tests/`
4. Run the full test suite: `pytest tests/`
5. Lint your code: `ruff check energykit/ tests/`
6. Format: `black energykit/ tests/`
7. Open a pull request against `main`

## Code style

- Python 3.9+ compatible
- `black` for formatting (line length 88)
- `ruff` for linting
- Type annotations encouraged but not mandatory
- scikit-learn API conventions where applicable (`fit` / `transform` / `predict`)
- All public classes and functions need docstrings

## Priority contribution areas

| Area | Description |
|------|-------------|
| 📊 Dataset loaders | ENTSO-E SFTP, PJM, AEMO, EIA API, OpenEI |
| 🔮 Forecasting | Neural models (N-BEATS, TFT, PatchTST), probabilistic intervals |
| ⚡ Disaggregation | LSTM-NILM, Seq2Seq, Transformer-based NILM |
| 🔋 Optimization | V2G bidirectional, VPP aggregation, BESS degradation model |
| 📐 Benchmarks | Standard benchmark datasets, leaderboard |
| 📖 Docs | Tutorials, API reference, Sphinx config |

## Reporting issues

Please use the [GitHub issue tracker](https://github.com/muranai/energykit/issues). Include:
- energykit version (`energykit.__version__`)
- Python version
- A minimal reproducible example

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
