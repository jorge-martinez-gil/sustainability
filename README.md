# Sustainability of Machine Learning Models

[![CI](https://github.com/jorge-martinez-gil/sustainability/actions/workflows/ci.yml/badge.svg)](https://github.com/jorge-martinez-gil/sustainability/actions/workflows/ci.yml)

A research-oriented Python project for studying **sustainability-aware machine learning model design** using multi-objective genetic programming (GP). The optimization jointly targets:

- **Prediction quality** (test MSE)
- **Model simplicity** (tree size)
- **Computational energy proxy** (CPU processing time)

## Motivation

Sustainable AI requires balancing predictive performance with computational efficiency and interpretability. This repository provides a compact but extensible baseline for experiments where these objectives conflict.

## Installation

```bash
python -m pip install -e .
```

For development (tests, linting, type checks):

```bash
python -m pip install -e .[dev]
```

## Quickstart

Run the default experiment:

```bash
python -m sustainability
```

Run with reduced settings for fast iteration:

```bash
python -m sustainability --generations 3 --population-size 60 --mu 40 --lambda 60 --quiet
```

## Reproducible workflow

1. Create and activate a virtual environment.
2. Install editable package with dev extras.
3. Run quality checks:

```bash
ruff check .
mypy src
pytest
```

4. Execute a deterministic run using a fixed seed:

```bash
python -m sustainability --seed 42 --quiet
```

### Expected outputs

The CLI prints:

- Best evolved symbolic expression
- Fitness tuple in the form `(simplicity, energy, test MSE)`

Exact numeric values can vary across hardware because the energy objective uses CPU process time.

## Project structure

```text
src/sustainability/      Python package and CLI
examples/                Runnable examples
tests/                   Automated tests
.github/workflows/       CI automation
```

## Citation

If you use this repository in academic work, please cite it using `CITATION.cff` metadata.

BibTeX (related foundational article):

```bibtex
@article{martinez2022sustainable,
  title={Sustainable semantic similarity assessment},
  author={Martinez-Gil, Jorge and Chaves-Gonzalez, Jose Manuel},
  journal={Journal of Intelligent \& Fuzzy Systems},
  volume={43},
  number={5},
  pages={6163--6174},
  year={2022},
  publisher={IOS Press}
}
```

## Roadmap

- Expand sustainability metrics beyond CPU time proxy
- Add benchmark datasets and experiment tracking
- Provide visualization notebooks for Pareto-front analysis

## License

MIT License. See [LICENSE](LICENSE).
