import numpy as np

from sustainability.core import make_example_dataset, run_experiment, safe_div


def test_safe_div_zero_returns_one() -> None:
    assert safe_div(1.0, 0.0) == 1.0


def test_make_example_dataset_is_deterministic() -> None:
    first_train, first_test = make_example_dataset(seed=42)
    second_train, second_test = make_example_dataset(seed=42)

    np.testing.assert_array_equal(first_train[0], second_train[0])
    np.testing.assert_array_equal(first_train[1], second_train[1])
    np.testing.assert_array_equal(first_test[0], second_test[0])
    np.testing.assert_array_equal(first_test[1], second_test[1])


def test_run_experiment_smoke() -> None:
    result = run_experiment(population_size=30, mu=20, lambda_=30, generations=1, verbose=False)

    assert len(result.fitness) == 3
    assert result.fitness[0] >= 1
    assert result.fitness[1] >= 0
    assert result.fitness[2] >= 0
