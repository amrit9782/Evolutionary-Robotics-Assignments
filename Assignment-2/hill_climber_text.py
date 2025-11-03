import random

# Target string
TARGET_STRING = "charles darwin was always seasick"
TARGET_LENGTH = len(TARGET_STRING)
CHARACTERS = "abcdefghijklmnopqrstuvwxyz "

# fitness function 
def calculate_fitness(candidate: str) -> int:
    '''Calculates the fitness by comparing the candidate string to the target string, character by character.'''
    if len(candidate) != TARGET_LENGTH:
        raise ValueError(f"Candidate string length {len(candidate)} does not match target length {TARGET_LENGTH}")
    score = 0
    for i in range(TARGET_LENGTH):
        if candidate[i] == TARGET_STRING[i]:
            score += 1
    return score

def mutate(parent_string: str) -> str:
    '''Mutates a single character in the parent string to a new, different character.'''
    index_to_mutate = random.randrange(TARGET_LENGTH)
    original_char = parent_string[index_to_mutate]
    
    new_char = original_char
    while new_char == original_char:
        new_char = random.choice(CHARACTERS)

    return parent_string[:index_to_mutate] + new_char + parent_string[index_to_mutate+1:]

def random_string(length: int) -> str:
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

        new_string = mutate(current_string)
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
        generation = run_simulation()
        generations_counts.append(generation)
        if (i + 1) % 10 == 0: 
             print(f"Completed run {i+1}/{num_runs}")
             
    # Calculate average generations
    average_generations = sum(generations_counts) / num_runs

    # Print results
    print(f"\n--- Results ---")
    print(f"Number of runs: {num_runs}")
    print(f"Average generations needed: {average_generations:.2f}")