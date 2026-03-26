# Contributing to Neural FDM Extended

Thank you for your interest in contributing. This project extends the Neural
FDM paper (Pastrana et al., ICLR 2025) with a variational autoencoder, GNN
encoder, and interactive design tools.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

Open an issue with:
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Full traceback if applicable

### Suggesting Features

Open an issue describing:
- The problem you want to solve
- Proposed solution
- Whether it stays within the paper's validated scope or is an extension

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Make your changes
5. Add tests for new functionality
6. Run the test suite: `pytest tests/ -v`
7. Run benchmarks: `python benchmarks/verify_all.py`
8. Submit a PR with a clear description

### Scope

Contributions should:

- **Stay within the paper's validated scope** when claiming paper-validated results
- **Clearly label extensions** that go beyond the paper (e.g., VAE, GNN)
- **Include tests** for any new functionality
- **Include references** to papers or equations in docstrings
- **Not break existing tests** (all 76 tests must pass)

## Development Setup

```bash
git clone https://github.com/efradeca/vae-fdm.git
cd vae-fdm
pip install -e ".[dev]"
pytest tests/ -v
```

## Code Style

- Python 3.10+
- Formatting: `ruff check src/ tests/`
- Docstrings: NumPy style with equation references
- Type hints where practical
- No code that appears AI-generated without review

## Testing

```bash
pytest tests/ -v                              # 76 unit tests
python benchmarks/verify_all.py               # 11 verification checks
python benchmarks/validation_suite.py         # 13 triple validation checks
python benchmarks/reproduce_paper.py          # Paper Table 1 reproduction
```

## Architecture

The codebase separates concerns:

- `src/neural_fdm/models.py` — Core encoder/decoder (paper, do not modify lightly)
- `src/neural_fdm/variational.py` — VAE extension (our contribution)
- `src/neural_fdm/gnn.py` — GNN encoder (our contribution)
- `src/neural_fdm/helpers.py` — FDM math (paper, do not modify)
- `scripts/` — Entry points and interactive tools
- `tests/` — Unit tests (pytest)
- `benchmarks/` — Validation and comparison suites
