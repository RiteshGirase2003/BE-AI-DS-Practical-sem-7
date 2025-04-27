import random
from deap import base, creator, tools, algorithms

# --- Problem Constants ---
IND_SIZE = 10  # Number of genes (dimensions)
LOW, UP = -5.12, 5.12  # Range of gene values

# --- DEAP Setup ---
# Define the fitness function (Minimization problem)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Create the individual class based on a list with fitness attribute
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator (genes)
toolbox.register("attr_float", random.uniform, LOW, UP)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fitness function: Sphere function
def evalSphere(individual):
    return sum(x**2 for x in individual),  # Return as tuple

toolbox.register("evaluate", evalSphere)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- Evolution Parameters ---
POP_SIZE = 100
NGEN = 50
CXPB = 0.5  # Crossover probability
MUTPB = 0.2  # Mutation probability

def main():
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: sum(f[0] for f in fits) / len(fits))
    stats.register("min", lambda fits: min(f[0] for f in fits))

    # Run the algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
                                   stats=stats, halloffame=hof, verbose=True)

    print("\nBest Individual:", hof[0])
    print("Fitness Value:", hof[0].fitness.values[0])

if __name__ == "__main__":
    main()