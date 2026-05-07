"""Minimal reproducible example for the sustainability package."""

from sustainability.core import run_experiment

if __name__ == "__main__":
    result = run_experiment(population_size=40, mu=20, lambda_=40, generations=3, verbose=False)
    print("Best individual:", result.best_individual)
    print("Fitness:", result.fitness)
