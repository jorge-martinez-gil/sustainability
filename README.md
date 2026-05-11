<div align="center">

# 🌱 Sustainability of Machine Learning Models

**A research-oriented Python project for building greener, smarter AI.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/jorge-martinez-gil/sustainability/ci.yml?style=for-the-badge&label=CI&logo=github)](https://github.com/jorge-martinez-gil/sustainability/actions)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-orange?style=for-the-badge)](https://github.com/astral-sh/ruff)

<br/>

> *Sustainable AI requires balancing predictive performance with computational efficiency and interpretability.*  
> This repository provides a compact but extensible baseline for multi-objective, sustainability-aware machine learning experiments.

</div>

---

## ✨ Key Features

| 🎯 Objective | 📐 Metric | Description |
|---|---|---|
| **Prediction Quality** | Test MSE | How accurately the model generalizes |
| **Model Simplicity** | Tree Size | Smaller models are more interpretable |
| **Energy Efficiency** | CPU Time Proxy | Lower compute cost = lower carbon footprint |

Multi-objective **Genetic Programming (GP)** is used to jointly optimize all three, exploring the Pareto front between performance and sustainability.

---

## 🚀 Getting Started

### Installation

```bash
# Standard install
python -m pip install -e .

# Development install (tests, linting, type checks)
python -m pip install -e .[dev]
```

### ⚡ Quickstart

Run the default experiment:

```bash
python -m sustainability
```

Run with reduced settings for fast iteration:

```bash
python -m sustainability --generations 3 --population-size 60 --mu 40 --lambda 60 --quiet
```

Run a fully deterministic experiment with a fixed seed:

```bash
python -m sustainability --seed 42 --quiet
```

---

## 🔬 Reproducible Workflow

```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install with dev extras
python -m pip install -e .[dev]

# 3. Run quality checks
ruff check .
mypy src
pytest

# 4. Execute a deterministic run
python -m sustainability --seed 42 --quiet
```

### 📊 Expected Outputs

The CLI prints:

- 🌳 **Best evolved symbolic expression**
- 📈 **Fitness tuple** in the form `(simplicity, energy, test MSE)`

> **Note:** Exact numeric values may vary across hardware because the energy objective uses CPU process time as a proxy.

---

## 🗂️ Project Structure

```
sustainability/
├── src/sustainability/      # Python package & CLI entry point
├── examples/                # Runnable example scripts
├── tests/                   # Automated test suite
└── .github/workflows/       # CI/CD automation
```

---

## 📚 Citation

If you use this repository in your academic work, please cite it using the `CITATION.cff` metadata.

<details>
<summary>📖 BibTeX — Related Foundational Article</summary>

```bibtex
@article{martinez2022sustainable,
  title     = {Sustainable semantic similarity assessment},
  author    = {Martinez-Gil, Jorge and Chaves-Gonzalez, Jose Manuel},
  journal   = {Journal of Intelligent \& Fuzzy Systems},
  volume    = {43},
  number    = {5},
  pages     = {6163--6174},
  year      = {2022},
  publisher = {IOS Press}
}
```

</details>

---

## 🗺️ Roadmap

- [ ] 🔋 Expand sustainability metrics beyond CPU time proxy
- [ ] 📦 Add benchmark datasets and experiment tracking
- [ ] 📉 Provide visualization notebooks for Pareto-front analysis
- [ ] 🤝 Integrate with carbon-aware computing frameworks

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made in the name of sustainable AI &nbsp;|&nbsp; [Jorge Martinez-Gil](https://github.com/jorge-martinez-gil)

</div>
