import pygame
import math 
import matplotlib.pyplot as plt
import numpy as np

pygame.init()
pygame.font.init()

# Setup
arena_size = 600
wall_thickness = 10
screen = pygame.display.set_mode((arena_size, arena_size))
clock = pygame.time.Clock()
pygame.display.set_caption("Proximity Sensor Vehicle Simulation")

# Walls
walls = [
    pygame.Rect(0, 0, arena_size, wall_thickness),                              # Top
    pygame.Rect(0, 0, wall_thickness, arena_size),                              # Left
    pygame.Rect(0, arena_size - wall_thickness, arena_size, wall_thickness),     # Bottom
    pygame.Rect(arena_size - wall_thickness, 0, wall_thickness, arena_size),     # Right 
    # Obstacles
    pygame.Rect(0, 0.7 * arena_size, 0.3 * arena_size, wall_thickness),
    pygame.Rect(0.7 * arena_size, 0.4 * arena_size, 0.3 * arena_size, wall_thickness),
    pygame.Rect(0.5 * arena_size, 0, wall_thickness, 0.4 * arena_size),
    pygame.Rect(0.5 * arena_size, 0.6 * arena_size, wall_thickness, 0.4 * arena_size),
]

# Robot State
x, y = 200, 500              # position
heading = 0.0                # radians; 0 points right (X-axis)
speed = 2.0                  # pixels per frame
turn_rate = 0.05             # radians per frame
robot_radius = 15
robot_color = (0, 150, 255)

trajectory = []  # stores robot positions for plotting

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
        for wall in walls:
            if wall.colliderect(point_rect):
                # Wall detected → normalize and return
                return 1 - d / sensor_range

    return 0.0  # no wall detected

# main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Autonomous Control based on Sensors ---
    sensor_readings = []
    sensor_range = 90
    sensor_angles = [-math.pi / 6, 0, math.pi / 6]  # left, center, right

    for a in sensor_angles:
        sensor_angle = heading + a
        value = cast_sensor_ray(x, y, sensor_angle, walls, sensor_range)
        sensor_readings.append(value)

    left_s, center_s, right_s = sensor_readings

    # --- Rule-based behavior ---
    turn_rate = 0.05
    base_speed = 2.0
    rotation = 0.0

    # control logic
    if center_s > 0.4:
        # Obstacle straight ahead - turn away depending on stronger side
        if left_s > right_s + 0.2:
            rotation = turn_rate      # turn right
        else:
            rotation = -turn_rate     # turn left
        speed = -1.5                  # slight reverse
    elif left_s > right_s + 0.2:
        rotation = turn_rate          # wall on left - turn right
        speed = base_speed
    elif right_s > left_s + 0.2:
        rotation = -turn_rate         # wall on right - turn left
        speed = base_speed
    else:
        rotation = 0.0
        speed = base_speed            # go straight

    # --- Update robot state ---
    heading += rotation
    x += speed * math.cos(heading)
    y += speed * math.sin(heading)

    # Record position
    trajectory.append((x, y, heading))

    # --- Keep robot inside boundaries ---
    left_bound = wall_thickness + robot_radius
    right_bound = arena_size - wall_thickness - robot_radius
    top_bound = wall_thickness + robot_radius
    bottom_bound = arena_size - wall_thickness - robot_radius

    x = max(left_bound, min(x, right_bound))
    y = max(top_bound, min(y, bottom_bound))

    # --- Draw Arena ---
    screen.fill((30, 30, 30))   # background
    for wall in walls:
        pygame.draw.rect(screen, (200, 200, 200), wall)                         # draw walls
    pygame.draw.circle(screen, robot_color, (int(x), int(y)), robot_radius)     # Draw Robot
    
    # --- Sensor Visualization and Reading ---
    sensor_readings = []
    sensor_range = 90
    sensor_angles = [-math.pi / 6, 0, math.pi / 6]

    for a in sensor_angles:
        # Compute absolute angle for each sensor
        sensor_angle = heading + a

        # Get sensor value (0–1)
        value = cast_sensor_ray(x, y, sensor_angle, walls, sensor_range)
        sensor_readings.append(value)

        # Compute ray endpoint for drawing
        end_x = x + sensor_range * math.cos(sensor_angle)
        end_y = y + sensor_range * math.sin(sensor_angle)

        # Draw ray
        pygame.draw.line(screen, (255, 0, 0), (x, y), (end_x, end_y), 1)

        # Draw a filled circle scaled by sensor value
        if value > 0:
            hit_x = x + (1 - value) * sensor_range * math.cos(sensor_angle)
            hit_y = y + (1 - value) * sensor_range * math.sin(sensor_angle)
            pygame.draw.circle(screen, (255, 255, 0), (int(hit_x), int(hit_y)), 4)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

# Plot the trajectory
traj_x, traj_y, traj_h = zip(*trajectory)

# Plot arena walls as rectangles
fig, ax = plt.subplots()
for wall in walls:
    rect = plt.Rectangle((wall.left, wall.top), wall.width, wall.height,
                         color='gray', alpha=0.7)
    ax.add_patch(rect)

# Plot robot trajectory
ax.plot(traj_x, traj_y, color='blue', linewidth=2, label='Robot Path')
ax.scatter(traj_x[0], traj_y[0], color='green', s=50, label='Start')
ax.scatter(traj_x[-1], traj_y[-1], color='red', s=50, label='End')

# Heading
N = 75 # Plot one arrow every 75 points
xs_sub = traj_x[::N]
ys_sub = traj_y[::N]
hs_sub = traj_h[::N]

# Calculate components
u = np.cos(hs_sub)
v = np.sin(hs_sub)

# Draw the arrows
ax.quiver(xs_sub, ys_sub, u, v,
          color='darkred',
          scale=40,
          headwidth=5,
          width=0.005,
          angles='xy',
          zorder=5)

# Set plot limits and labels
ax.set_aspect('equal')
ax.set_xlim(0, arena_size)
ax.set_ylim(0, arena_size)
ax.set_title("Task 2: Proximity sensor Robot Trajectory")
ax.set_xlabel("X position (pixels)")
ax.set_ylabel("Y position (pixels)")
ax.invert_yaxis()
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
plt.subplots_adjust(right=0.75)         # Adjust the subplot to make room for the legend on the right
plt.savefig("./Assignments/1/proximity_trajectory_plot.png", dpi=600)
plt.show()