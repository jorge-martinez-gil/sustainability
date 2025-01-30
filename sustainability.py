# This script demonstrates how to optimize a GP model for energy consumption, simplicity, and accuracy.
# The script uses the DEAP library to define the problem, create the GP algorithm, and run the optimization process.

import operator
import random
import numpy as np
from deap import base, creator, gp, tools, algorithms
import time
from sklearn.model_selection import train_test_split

# Define a safe division function to avoid division by zero
def safe_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

# Example dataset
X = np.linspace(-1, 1, 100)
y = X ** 2 + np.sin(3 * X)  # Example target function

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42)
train_dataset = (X_train, y_train)
test_dataset = (X_test, y_test)

# Evaluation function (without including train MSE in the return)
def eval_func(individual, toolbox, dataset, test_dataset):
    func = toolbox.compile(expr=individual)
    X_train, y_train = dataset
    X_test, y_test = test_dataset
    
    # --- Training MSE computed but not used as an objective ---
    train_predictions = np.array([func(x) for x in X_train])
    train_mse = np.mean((train_predictions - y_train) ** 2)
    
    # Generalization (Test accuracy)
    test_predictions = np.array([func(x) for x in X_test])
    test_mse = np.mean((test_predictions - y_test) ** 2)
    
    # Simplicity (tree length)
    simplicity = len(individual)
    
    # Energy consumption
    start_time = time.process_time()
    for _ in range(100):
        np.array([func(x) for x in X_train])
    energy = time.process_time() - start_time
    
    # Return only the three objectives: simplicity, energy, test_mse
    return simplicity, energy, test_mse

# Problem definition
def setup_toolbox(dataset):
    pset = gp.PrimitiveSet("MAIN", 1)  # 1 input variable (X)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(safe_div, 2)  # Use safe division
    pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))
    pset.renameArguments(ARG0="x")
    
    # Define fitness for 3 objectives (all to be minimized)
    # We have: (simplicity, energy, test_mse)
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", eval_func, toolbox=toolbox, 
                     dataset=train_dataset, test_dataset=test_dataset)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=17))
    return toolbox

# Run the genetic programming algorithm
toolbox = setup_toolbox(train_dataset)
population = toolbox.population(n=300)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min, axis=0)
stats.register("avg", np.mean, axis=0)

population, logbook = algorithms.eaMuPlusLambda(
    population, toolbox, mu=200, lambda_=300, 
    cxpb=0.6, mutpb=0.3, ngen=40, 
    stats=stats, halloffame=hof, verbose=True
)

# Best solution
print("Best individual:", hof[0])
print("Fitness (simplicity, energy, test MSE):", hof[0].fitness.values)