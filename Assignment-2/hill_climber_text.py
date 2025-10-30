import random 

# Target string
TARGET_STRING = "charles darwin was always seasick"
TARGET_LENGTH = len(TARGET_STRING)
CHARACTERS = "abcdefghijklmnopqrstuvwxyz "

# fitness function 
def calculate_fitness(candidate):
    '''Calculates the fitness by comparing the candidate string to the target string, character by character.'''
    score = 0
    for i in range(TARGET_LENGTH):
        if candidate[i] == TARGET_STRING[i]:
            score += 1
    return score

def random_string(length):
    '''Generates a random string of given length'''
    return "".join(random.choice(CHARACTERS) for _ in range(length))

def run_simulation():
    '''Runs the hill climbing simulation to evolve a string towards the target string.'''
    generation = 0                                          # start generation at 0
    current_string = random_string(TARGET_LENGTH)           # generate initial random string
    current_fitness = calculate_fitness(current_string)     # calculate fitness of initial string

    # print(f'Generation, fitness, string')
    # Hill climbing algorithm
    while current_fitness < TARGET_LENGTH:
        # print(f"#{generation} {current_fitness*100/TARGET_LENGTH:.0f}% {current_string}")

        generation += 1                                 # increment generation count

        # Mutate the current string
        index_to_mutate = random.randint(0, TARGET_LENGTH - 1)  # choose a random index to mutate
        new_char = random.choice(CHARACTERS)                    # choose a new random character
        new_string_list = list(current_string)                  # create a new mutated string
        new_string_list[index_to_mutate] = new_char
        new_string = "".join(new_string_list)

        # Evaluate
        new_fitness = calculate_fitness(new_string)

        # Selection
        # Accept the new string if it's better
        if new_fitness >= current_fitness:
            current_string = new_string
            current_fitness = new_fitness

    return generation

# Main execution
if __name__ == "__main__":
    num_runs = 1000
    generations_counts = []
    print(f"Running {num_runs} simulations...")

    for i in range(num_runs):
        generations = run_simulation()
        generations_counts.append(generations)
        if (i + 1) % 10 == 0: 
             print(f"Completed run {i+1}/{num_runs}")
             
    # Calculate average generations
    average_generations = sum(generations_counts) / num_runs

    # Print results
    print(f"\n--- Results ---")
    print(f"Number of runs: {num_runs}")
    print(f"Average generations needed: {average_generations:.2f}")