"""Core experiment logic for multi-objective GP sustainability optimization."""

from __future__ import annotations

import operator
import random
import time
from dataclasses import dataclass

import numpy as np
from deap import algorithms, base, creator, gp, tools
from sklearn.model_selection import train_test_split

ArrayPair = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class GPResult:
    """Container for the best model and optimization metadata."""

    best_individual: gp.PrimitiveTree
    fitness: tuple[float, float, float]


def safe_div(left: float, right: float) -> float:
    """Divide values while avoiding division-by-zero failures."""
    try:
        return left / right
    except ZeroDivisionError:
        return 1.0


def random_ephemeral_constant() -> float:
    """Generate ephemeral constants used by the GP primitives."""
    return random.uniform(-1.0, 1.0)


def make_example_dataset(
    *,
    n_samples: int = 100,
    test_size: float = 0.3,
    seed: int = 42,
) -> tuple[ArrayPair, ArrayPair]:
    """Create the synthetic dataset used in the sustainability experiments."""
    x = np.linspace(-1, 1, n_samples)
    y = x**2 + np.sin(3 * x)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
    )
    return (x_train, y_train), (x_test, y_test)


def eval_func(
    individual: gp.PrimitiveTree,
    toolbox: base.Toolbox,
    dataset: ArrayPair,
    test_dataset: ArrayPair,
) -> tuple[float, float, float]:
    """Evaluate a GP individual on simplicity, energy usage, and test MSE."""
    func = toolbox.compile(expr=individual)
    x_train, y_train = dataset
    x_test, y_test = test_dataset

    test_predictions = np.array([func(x) for x in x_test])
    test_mse = float(np.mean((test_predictions - y_test) ** 2))

    simplicity = float(len(individual))

    start_time = time.process_time()
    for _ in range(100):
        np.array([func(x) for x in x_train])
    energy = float(time.process_time() - start_time)

    return simplicity, energy, test_mse


def _ensure_creator_types() -> None:
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)


def setup_toolbox(dataset: ArrayPair, test_dataset: ArrayPair) -> base.Toolbox:
    """Prepare the DEAP toolbox for multi-objective GP evolution."""
    _ensure_creator_types()

    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(safe_div, 2)
    pset.addEphemeralConstant("rand", random_ephemeral_constant)
    pset.renameArguments(ARG0="x")

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register(
        "evaluate",
        eval_func,
        toolbox=toolbox,
        dataset=dataset,
        test_dataset=test_dataset,
    )
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=17))
    return toolbox


def run_experiment(
    *,
    seed: int = 42,
    population_size: int = 300,
    mu: int = 200,
    lambda_: int = 300,
    generations: int = 40,
    cxpb: float = 0.6,
    mutpb: float = 0.3,
    verbose: bool = True,
) -> GPResult:
    """Run the complete sustainability GP optimization experiment."""
    random.seed(seed)
    np.random.seed(seed)

    train_dataset, test_dataset = make_example_dataset(seed=seed)
    toolbox = setup_toolbox(train_dataset, test_dataset)

    population = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("avg", np.mean, axis=0)

    algorithms.eaMuPlusLambda(
        population,
        toolbox,
        mu=mu,
        lambda_=lambda_,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=verbose,
    )

    fitness_values = hof[0].fitness.values
    return GPResult(
        best_individual=hof[0],
        fitness=(float(fitness_values[0]), float(fitness_values[1]), float(fitness_values[2])),
    )
