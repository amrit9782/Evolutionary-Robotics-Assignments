# Ackley Optimization Problem using Evolutionary Algorithms
import numpy as np
import matplotlib.pyplot as plt

POPULATION_SIZE = 1000
BOUNDS = [-32.768, 32.768]
NUM_GENERATIONS = 1000

# Genetic Representation
def create_individuals(bounds):
    """Create POPULATION_SIZE random individuals within the given bounds."""
    return np.array([np.random.uniform(bounds[0], bounds[1], 3) for _ in range(POPULATION_SIZE)])

def evaluation_function(x, y, z):
    """Compute the Ackley function value for a given input 3D vector."""

    f = \
    (
        -20.0 * np.exp(
            -0.2 * np.sqrt( (1/3) * (x**2 + y**2 + z**2))
        )
        - np.exp(
            (1/3) * ( np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y) + np.cos(2 * np.pi * z) )
        )
        + 20 + np.exp(1)
    )
    return 1/(f + 1) # Invert to convert minimization to maximization

def run_generation(population):

    evaluations = np.array([evaluation_function(ind[0], ind[1], ind[2]) for ind in population])

    sorted_indices = np.argsort(evaluations)[::-1]  # Sort in descending order
    best_individuals = population[sorted_indices][:POPULATION_SIZE]

    # Replacement strategy: elitism (top X% individuals are kept without crossover/mutation for next generation)
    # keep first X% as elites
    num_elites = int(POPULATION_SIZE * 0.1)
    elites = best_individuals[:num_elites].copy()
    crossover_point = 1  # Crossover after the first gene
    for i in range(num_elites, POPULATION_SIZE, 2):  # Crossover last (worst) 100-X% individuals
        if i+1 >= POPULATION_SIZE:
            break
        parent1 = best_individuals[i]
        parent2 = best_individuals[i+1]
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        best_individuals[i] = child1
        best_individuals[i+1] = child2

    # Mutation of last (worst) 100-X% individuals
    mutation_rate = 0.2
    for ind in best_individuals[num_elites:]:
        for i in range(len(ind)):
            if np.random.rand() < mutation_rate:
                ind[i] = np.random.uniform(BOUNDS[0], BOUNDS[1])

    # Add back elites
    best_individuals[:num_elites] = elites

    best_evaluations = np.array([evaluation_function(ind[0], ind[1], ind[2]) for ind in best_individuals])
    sorted_indices = np.argsort(best_evaluations)[::-1]
    best_individuals = best_individuals[sorted_indices][:POPULATION_SIZE]
    best_evaluations = best_evaluations[sorted_indices][:POPULATION_SIZE]

    return best_individuals, best_evaluations[0], np.mean(best_evaluations)


if __name__ == "__main__":
    best_fittness_points = np.array([])
    avg_fittness_points = np.array([])
    population = create_individuals(BOUNDS)

    for i in range(NUM_GENERATIONS):
        population, best_fittness, avg_fittness = run_generation(population)
        best_fittness_points = np.append(best_fittness_points, best_fittness)
        avg_fittness_points = np.append(avg_fittness_points, avg_fittness)

    plt.plot(best_fittness_points, label='Best Fitness')
    plt.plot(avg_fittness_points, label='Average Fittness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Ackley Fn Optimization for {NUM_GENERATIONS} Generations and Population {POPULATION_SIZE}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig('./Assignment-3/ackley-fitness-plot.png', dpi=300)
    plt.show()

