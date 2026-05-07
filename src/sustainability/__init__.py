"""Tools for studying sustainability of machine learning models with genetic programming."""

from .core import (
    GPResult,
    eval_func,
    make_example_dataset,
    run_experiment,
    safe_div,
    setup_toolbox,
)

__all__ = [
    "GPResult",
    "eval_func",
    "make_example_dataset",
    "run_experiment",
    "safe_div",
    "setup_toolbox",
]
