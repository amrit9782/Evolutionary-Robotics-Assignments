import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Hyperparameters ---
DATA_FILE = './Assignment-3/data'
POP_SIZE = 50               # Population size
GENOME_LENGTH = 3           # Three weights: [w0, w1, w2]
GENERATIONS = 100           # Number of generations to run
MUTATION_RATE = 0.1         # Chance for each gene to mutate
MUTATION_STRENGTH = 0.5     # Std. dev. of mutation
ELITISM_SIZE = 2            # Keep the 2 best individuals

# functions
# -- load data --
def load_data(filename):
    """ Loads the classification file """
    data = pd.read_csv(filename,
                       delim_whitespace=True,
                       header=None,
                       names=['index', 'class', 'x', 'y'])

    # clean the data, drop any rows with NaN
    initial_count = len(data)
    data = data.dropna()
    final_count = len(data)

    if initial_count > final_count:
        print(f"Cleaned data: {initial_count - final_count} rows with missing values removed.")
    
    # Set class dtype as int
    data['class'] = data['class'].astype(int)

    return data

# -- ANN Output --
def ann_output(genome, x, y):
    ''' Calculate output of a simple ANN '''
    w0, w1, w2 = genome
    net_input = w0 * 1 + w1 * x + w2 * y

    # Activation function: phi(x) = 2 / (1 + exp(-2x)) - 1
    output = (2 / (1 + np.exp(-2 * net_input))) - 1

    return output

# -- fitness function --
def calc_fitness(genome, data_x, data_y, data_class):
    """ Calculate the fitness of a single genome
     by comparing with the correct number of predicted classes """
    ann_outputs = ann_output(genome, data_x, data_y)
    predicted_classes = np.where(ann_outputs < 0, 0, 1)

    correct_count = np.sum(predicted_classes == data_class)
    return correct_count

# -- EA components --
def initial_population(pop_size, genome_length):
    ''' Initialize the population with random weights'''
    return np.random.uniform(-5.0, 5.0, (pop_size, genome_length))

def selection(population, fitness_score):
    ''' performs tournament selection '''
    tournament_size = 3

    # Pick 3 random individuals from the population
    contender_indices = np.random.randint(0, len(population), tournament_size)
    contender_fitnesses = fitness_score[contender_indices]

    # find the winner, highest fitness
    winner_local_index = np.argmax(contender_fitnesses)
    winner_global_index = contender_indices[winner_local_index]

    return population[winner_global_index]

def mutation(genome, mutation_rate, mutation_strength):
    ''' Applies mutation to a genome '''
    mutated_genome = genome.copy()
    for i in range(len(mutated_genome)):
        if np.random.rand() < mutation_rate:
            mutation_val = np.random.normal(0 , mutation_strength)
            mutated_genome[i] += mutation_val

    return mutated_genome

# -- plotting --
def plot_fitness_progress(log_best_fitness, log_avg_fitness):
    ''' Plot the best and average fitness '''
    plt.figure()
    plt.plot(log_best_fitness, label='Best fitness')
    plt.plot(log_avg_fitness, label='Average fitness')
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Correct Classifications)")
    plt.title("Evolutionary Algorithm Fitness Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig("./Assignment-3/ex2/ex2_fitness_plot.png")
    plt.show()

def plot_classifier(best_genome, data):
    ''' plot the data points & final evolved fitness '''
    w0, w1, w2 = best_genome

    plt.figure()

    # Plot the two classes of data points
    class_0 = data[data['class'] == 0]
    class_1 = data[data['class'] == 1]

    plt.scatter(class_0['x'], class_0['y'],
                label="Class 0", marker='o', alpha=0.7)
    
    plt.scatter(class_1['x'], class_1['y'],
                label="Class 1", marker='s', alpha=0.7)

    # Calculate and plot the decision boundary line
    # The line is where net_input = 0: w0 + w1*x + w2*y = 0
    # Solving for y gives: y = (-w1/w2)*x - (w0/w2)
    
    # Get x values across the plot
    x_min, x_max = data['x'].min(), data['x'].max()
    plot_x = np.array([x_min, x_max])
    
    # Calculate corresponding y values
    if w2 != 0: # Avoid division by zero
        plot_y = (-w1 / w2) * plot_x - (w0 / w2)
        plt.plot(
            plot_x, plot_y,
            color='r', linestyle='--', label="Evolved Decision Boundary"
        )
    
    plt.xlabel("Feature x")
    plt.ylabel("Feature y")
    plt.title("Data and Evolved Classifier")
    plt.legend()
    plt.grid(True)
    plt.savefig("./Assignment-3/ex2/ex2_classifier_plot.png")
    plt.show()

# main loop
def main():
    '''
    Main to run the EA
    '''
    # load data
    data = load_data(DATA_FILE)
    if data is None:
        return
    
    # Total possible fitness
    max_fitness = len(data)

    data_x = data['x'].values
    data_y = data['y'].values
    data_class = data['class'].values

    # initialize population
    population = initial_population(POP_SIZE, GENOME_LENGTH)

    # Run the EA loop
    print("\n--- Starting Evolution ---")
    log_best_fitness = []
    log_avg_fitness = []
    best_genome_overall = None
    best_fitness_overall = -1

    for gen in range(GENERATIONS):
        # calculate the fitness for all individuals
        fitness_scores = np.array([
            calc_fitness(ind, data_x, data_y, data_class)
            for ind in population
        ])

        # Log metrics
        best_gen_fitness = np.max(fitness_scores)
        avg_gen_fitness = np.mean(fitness_scores)
        log_best_fitness.append(best_gen_fitness)
        log_avg_fitness.append(avg_gen_fitness)

        # Check for new best
        if best_gen_fitness > best_fitness_overall:
            best_fitness_overall = best_gen_fitness
            best_genome_overall = population[np.argmax(fitness_scores)].copy()
        
        if gen % 10 == 0:
            print(f"Gen {gen:3}: Best={best_gen_fitness}, Avg={avg_gen_fitness:.2f}")

        # Create the next generation
        new_population = []
        
        # Elitism: Copy the best N individuals
        elite_indices = np.argsort(fitness_scores)[-ELITISM_SIZE:]
        for i in elite_indices:
            new_population.append(population[i].copy())

        # Fill the rest of the population
        while len(new_population) < POP_SIZE:
            # Select a parent
            parent = selection(population, fitness_scores)
            
            # Create a child via mutation
            child = mutation(parent, MUTATION_RATE, MUTATION_STRENGTH)
            
            new_population.append(child)
            
        population = np.array(new_population)
        
    print("--- Evolution Finished ---")

    # Show results
    print("\n--- Final Results ---")
    print(f"Best Fitness Found: {best_fitness_overall} / {max_fitness} correct")
    print(f"Best Genome (weights [w0, w1, w2]):")
    print(f"  w0 (bias): {best_genome_overall[0]:.4f}")
    print(f"  w1 (x-weight): {best_genome_overall[1]:.4f}")
    print(f"  w2 (y-weight): {best_genome_overall[2]:.4f}")

    # --- 5. Plot Results ---
    plot_fitness_progress(log_best_fitness, log_avg_fitness)
    plot_classifier(best_genome_overall, data)

# Run the main function
if __name__ == "__main__":
    main()
