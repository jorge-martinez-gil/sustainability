"""CLI for running sustainability-focused GP experiments."""

from __future__ import annotations

import argparse

from .core import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sustainability-focused GP optimization.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--population-size", type=int, default=300, help="Population size.")
    parser.add_argument(
        "--mu", type=int, default=200, help="Number of survivors each generation."
    )
    parser.add_argument(
        "--lambda", dest="lambda_", type=int, default=300, help="Number of offspring."
    )
    parser.add_argument("--generations", type=int, default=40, help="Number of generations.")
    parser.add_argument("--quiet", action="store_true", help="Disable DEAP progress logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(
        seed=args.seed,
        population_size=args.population_size,
        mu=args.mu,
        lambda_=args.lambda_,
        generations=args.generations,
        verbose=not args.quiet,
    )
    print("Best individual:", result.best_individual)
    print("Fitness (simplicity, energy, test MSE):", result.fitness)


if __name__ == "__main__":
    main()
