import pygame
import math 
import matplotlib.pyplot as plt
import numpy as np
import random

# Setup
arena_size = 600
wall_thickness = 10

# Walls
walls = [
    pygame.Rect(0, 0, arena_size, wall_thickness),                              # Top
    pygame.Rect(0, 0, wall_thickness, arena_size),                              # Left
    pygame.Rect(0, arena_size - wall_thickness, arena_size, wall_thickness),     # Bottom
    pygame.Rect(arena_size - wall_thickness, 0, wall_thickness, arena_size),     # Right 
    # Obstacles
    pygame.Rect(0, int(0.7 * arena_size), int(0.3 * arena_size), wall_thickness),
    pygame.Rect(int(0.7 * arena_size), int(0.4 * arena_size), int(0.3 * arena_size), wall_thickness),
    pygame.Rect(int(0.5 * arena_size), 0, wall_thickness, int(0.4 * arena_size)),
    pygame.Rect(int(0.5 * arena_size), int(0.6 * arena_size), wall_thickness, int(0.4 * arena_size)),
]

grid_resolution = 0.01 * arena_size
simulation_steps = 1000

# --- Robot Constants ---
robot_radius = 15
robot_color = (0, 150, 255)
wheel_base = 2 * robot_radius       # Distance between wheels (pixels)
max_wheel_speed = 5.0               # Max pixels/frame for each wheel
min_wheel_speed = 0.0              # Min pixels/frame
sensor_range = 90
sensor_angles = [-math.pi / 6, 0, math.pi / 6]  # left, center, right

# Robot initial state
initial_x, initial_y = 200, 500 
initial_heading = 0.0 

x, y = initial_x, initial_y
heading = initial_heading

# Genome definition
# Genome = [m0, c0, m1, c1, m2, c2] for 3 sensors
def generate_random_genome():
    return [random.uniform(-5, 5) for _ in range(6)]

def get_wheel_speeds(genome, sensor_readings):
    '''Calculates the wheel speeds based on genome and sensor readings.'''
    m0, c0, m1, c1, m2, c2 = genome
    s_l, s_m, s_r = sensor_readings

    # raw speeds
    v_l_raw = m0 * s_l + c0
    v_r_raw = m1 * s_r + c1 + m2 * s_m + c2

    v_l = v_l_raw
    v_r = v_r_raw

    # Clamp speeds to min/max
    # v_l = max(min_wheel_speed, min(max_wheel_speed, v_l_raw))
    # v_r = max(min_wheel_speed, min(max_wheel_speed, v_r_raw))

    return v_l, v_r

# Proximity Sensor Ray Casting
def cast_sensor_ray(x, y, angle, walls, sensor_range):
    """
    Cast a ray from (x, y) at 'angle' and return the distance to the nearest wall
    within sensor_range. Returns sensor value in [0, 1].
    """
    step = 2  # how far we move per iteration along the ray
    for d in range(0, sensor_range, step):
        # Compute test point along the ray
        test_x = x + d * math.cos(angle)
        test_y = y + d * math.sin(angle)

        # Build a tiny point rectangle to check collision
        point_rect = pygame.Rect(test_x, test_y, 2, 2)

        wall_indices = point_rect.collidelistall(walls)
        if wall_indices: 
             return 1.0 - (d / sensor_range)

    return 0.0  # no wall detected

# Evaluation function
def evaluate_genome(genome, step_count=0, visualize=False):
    ''' Runs one simulation for a genome and returns the fitness'''

    # Reset robot state
    x, y = initial_x, initial_y
    heading = initial_heading

    # initialize fitness
    visited_cells = set()
    current_trajectory = []
    fitness_multiplier = 1.0
    displacement_bonus = 0.75

    start_x, start_y = initial_x, initial_y

    screen = None   
    clock = None
    if visualize:
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((arena_size, arena_size))
        clock = pygame.time.Clock()
        pygame.display.set_caption("Evaluating Genome...")

    # Run simulation loop for 1000 steps
    running = True
    step_count = step_count
    while running and step_count < simulation_steps:
        step_count += 1

        if visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if not running: break

        # Store previous position in case of collision
        prev_x, prev_y = x, y

        # --- Sensor Readings ---
        sensor_readings = []
        for a in sensor_angles:
            sensor_angle = heading + a
            value = cast_sensor_ray(x, y, sensor_angle, walls, sensor_range)
            sensor_readings.append(value)

        # Control based on genome
        v_l, v_r = get_wheel_speeds(genome, sensor_readings)

        # Robot kinematics
        speed = (v_l + v_r) / 2.0
        rotation = (v_r - v_l) / wheel_base

        # update robot state
        heading += rotation
        heading = (heading + math.pi) % ( 2 * math.pi) - math.pi    

        x += speed * math.cos(heading)
        y += speed * math.sin(heading)

        # Calculate Grid cell and update fitness
        cell_x = int(x // grid_resolution)
        cell_y = int(y // grid_resolution)
        visited_cells.add((cell_x, cell_y))

        # Record position
        current_trajectory.append((x, y, heading))

        # keep robot inside boundaries
        left_bound = wall_thickness + robot_radius
        right_bound = arena_size - wall_thickness - robot_radius
        top_bound = wall_thickness + robot_radius
        bottom_bound = arena_size - wall_thickness - robot_radius

        # Check for collisions with walls
        collided = False
        robot_rect = pygame.Rect(x - robot_radius, y - robot_radius, 2 * robot_radius, 2 * robot_radius)
        # Check all walls including boundaries
        if robot_rect.collidelist(walls) != -1: 
             collided = True

        if collided:
            # print("Collision detected! Penalizing the genome.") 
            fitness_multiplier *= 0.5
            # print(f"Collision! Multiplier now: {fitness_multiplier}")   
            x, y = prev_x, prev_y

        # Clamp position 
        x = max(left_bound, min(x, right_bound))
        y = max(top_bound, min(y, bottom_bound))

        # Store final position
        final_x, final_y = x, y

        if visualize:
            # --- Draw Arena ---
            screen.fill((30, 30, 30))   # background
            for wall in walls:
                pygame.draw.rect(screen, (200, 200, 200), wall)                         # draw walls
            pygame.draw.circle(screen, robot_color, (int(x), int(y)), robot_radius)     # Draw Robot
            
            # --- Sensor Visualization and Reading ---
            for i, a in enumerate(sensor_angles):
                sensor_angle = heading + a
                value = sensor_readings[i] 
                end_x = x + sensor_range * math.cos(sensor_angle)
                end_y = y + sensor_range * math.sin(sensor_angle)
                pygame.draw.line(screen, (255, 0, 0), (x, y), (end_x, end_y), 1)
                if value > 0:
                    hit_x = x + (1 - value) * sensor_range * math.cos(sensor_angle)
                    hit_y = y + (1 - value) * sensor_range * math.sin(sensor_angle)
                    pygame.draw.circle(screen, (255, 255, 0), (int(hit_x), int(hit_y)), 4)

            pygame.display.flip()
            clock.tick(60)

    if visualize:
        pygame.quit()
    
    # Calculate final fitness
    base_fitness = len(visited_cells)
    # collided_fitness = base_fitness * fitness_multiplier

    # displacement = math.sqrt((final_x - start_x)**2 + (final_y - start_y)**2)
    # displacement_bonus = displacement * displacement_bonus
    
    # final_fitness = collided_fitness + displacement_bonus

    return base_fitness, current_trajectory

if __name__ == "__main__":
    # hill climber params
    num_generations = 5000
    mutation_std_dev = 3.0

    # hill climber initialization
    current_genome = generate_random_genome()
    current_fitness, current_trajectory = evaluate_genome(current_genome, visualize=False)
    best_genome = current_genome
    best_fitness = current_fitness

    print(f"Generation 0/{num_generations}: Best Fitness = {best_fitness:.2f}")

    # Hill climbing loop
    for generation in range(1, num_generations + 1):
        # mutate
        new_genome = list(current_genome)
        index_to_mutate = random.randint(0, len(new_genome) - 1)
        noise = random.gauss(0, mutation_std_dev)
        new_genome[index_to_mutate] += noise

        # evaluate
        new_fitness, new_trajectory = evaluate_genome(new_genome, visualize=False)

        # selection
        if new_fitness >= current_fitness:
            current_genome = new_genome
            current_fitness = new_fitness
            current_trajectory = new_trajectory

            # update best genome found
            if current_fitness > best_fitness:
                best_genome = current_genome
                best_fitness = current_fitness
                print(f"Generation {generation}/{num_generations}: New Best Fitness = {best_fitness:.2f}")

        if generation % 50 == 0:
            print(f"Generation {generation}/{num_generations}: Current Fitness = {current_fitness:.2f} (Best Ever: {best_fitness:.2f})")
    
    # Final evaluation with visualization
    print("\nHill climbing finished.")
    print(f"Best genome found: {best_genome}")
    print(f"Best fitness achieved: {best_fitness}")

    print("\nEvaluating the best genome with visualization...")
        
    final_fitness, final_trajectory = evaluate_genome(best_genome, visualize=True)

    # plotting
    if final_trajectory:
        traj_x, traj_y, traj_h = zip(*final_trajectory)
        fig, ax = plt.subplots()
        for wall in walls:
            rect = plt.Rectangle((wall.left, wall.top), wall.width, wall.height, color='gray', alpha=0.7)
            ax.add_patch(rect)

        # Plot robot trajectory
        ax.plot(traj_x, traj_y, color='blue', linewidth=1, label='Robot Path')
        ax.scatter(traj_x[0], traj_y[0], color='green', s=30, label='Start', zorder=5)
        ax.scatter(traj_x[-1], traj_y[-1], color='red', s=30, label='End', zorder=5)

        # Heading Arrows (optional)
        N = 75
        if len(traj_x) > N: # Check if trajectory is long enough for subsampling
            xs_sub = traj_x[::N]
            ys_sub = traj_y[::N]
            hs_sub = traj_h[::N]
            u = np.cos(hs_sub)
            v = np.sin(hs_sub)
            ax.quiver(xs_sub, ys_sub, u, v,
                    color='darkred', scale=40, headwidth=5, width=0.005, angles='xy', zorder=5)
            
        ax.set_aspect('equal')
        ax.set_xlim(0, arena_size)
        ax.set_ylim(0, arena_size)
        ax.set_title(f"Task 2: Best Robot Trajectory (Fitness: {best_fitness})")
        ax.set_xlabel("X position (pixels)")
        ax.set_ylabel("Y position (pixels)")
        ax.invert_yaxis() 
        ax.legend()
        
        # Save the plot (update path as needed)
        plot_filename = "./Assignment-2/hill_climber_proximity_best_trajectory.png"
        plt.savefig(plot_filename, dpi=300)
        print(f"Trajectory plot saved as {plot_filename}")
        plt.show()
    else:
        print("No trajectory recorded for the best genome.")