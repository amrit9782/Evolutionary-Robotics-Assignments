import pygame
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.patches as mpatches

# --- Initialization of Pygame and fonts ---
pygame.init()
pygame.font.init()

# Get screen size dynamically to adapt simulation size
infoObject = pygame.display.Info()
WIDTH, HEIGHT = infoObject.current_w, infoObject.current_h

# Constants for simulation
FPS = 60  # Frames per second

# Light source parameters
LIGHT_POS = [WIDTH // 2, HEIGHT // 2]  # Start light source in center
LIGHT_RADIUS = 400  # Radius of light influence

# Robot parameters
ROBOT_RADIUS = 15  # Robot radius for drawing
SENSOR_OFFSET_ANGLE = math.pi / 4  # Sensor offset angle (±45°)
SENSOR_OFFSET_DIST = 20  # Distance of sensors from robot center
C = 0.05  # Turning sensitivity coefficient

# Setup Pygame window and clock
win = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 18)

# Initial robot state variables
x, y = 0, 0  # Start position top-left
heading = 3 * math.pi / 2  # Facing upward direction (270 degrees radians)
speed = 0  # Initial speed
angular_speed = 0  # Initial angular speed
max_speed = 3  # Max linear speed
max_angular_speed = 0.1  # Max angular speed

# Simulation mode flag: Aggressor or Fear mode
is_aggressor = True

# Button to toggle modes placed at top-right
button_width, button_height = 220, 30
button_rect = pygame.Rect(WIDTH - button_width - 10, 10, button_width, button_height)

# --- Utility function definitions ---

def light_intensity(pos):
    """
    Calculate light intensity at position pos based on distance to light source.
    Intensity drops linearly to zero at LIGHT_RADIUS distance.
    """
    dx = pos[0] - LIGHT_POS[0]
    dy = pos[1] - LIGHT_POS[1]
    dist = math.sqrt(dx * dx + dy * dy)
    intensity = max(0, 1 - dist / LIGHT_RADIUS)
    return intensity

def sensor_position(x, y, heading, offset_angle):
    """
    Calculate the (x, y) position of a sensor relative to the robot's center position and heading.
    offset_angle determines the angular offset of the sensor from the robot's facing direction.
    """
    angle = heading + offset_angle
    sx = x + SENSOR_OFFSET_DIST * math.cos(angle)
    sy = y + SENSOR_OFFSET_DIST * math.sin(angle)
    return sx % WIDTH, sy % HEIGHT  # Wrap within screen bounds

def split_trajectory_on_wrap(xy, width, height):
    """
    Splits a trajectory into segments whenever a wrap-around jump happens.
    This prevents drawing erroneous lines across screen edges.
    """
    if len(xy) == 0:
        return []
    segments = []
    segment = [xy[0]]
    for i in range(1, len(xy)):
        prev = xy[i-1]
        curr = xy[i]
        dx = abs(curr[0] - prev[0])
        dy = abs(curr[1] - prev[1])
        # Detect wrap jump by large distance on x or y
        if dx > width / 2 or dy > height / 2:
            segments.append(np.array(segment))
            segment = [curr]
        else:
            segment.append(curr)
    if segment:
        segments.append(np.array(segment))
    return segments

# --- List to log robot states during simulation ---
robot_traj = []  # Each entry: [x, y, mode, manual_control_flag]

# --- Main simulation loop ---
running = True
while running:
    # Handle input events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos):
                # Toggle mode when button clicked
                is_aggressor = not is_aggressor
            else:
                # Move light source position when clicked elsewhere
                LIGHT_POS[0], LIGHT_POS[1] = event.pos

    keys = pygame.key.get_pressed()
    manual_control = any(keys)  # True if any control keys pressed

    # Calculate sensor positions and light intensities
    sl_pos = sensor_position(x, y, heading, SENSOR_OFFSET_ANGLE)
    sr_pos = sensor_position(x, y, heading, -SENSOR_OFFSET_ANGLE)
    sl = light_intensity(sl_pos)
    sr = light_intensity(sr_pos)

    # Log robot state
    robot_traj.append([x, y, is_aggressor, manual_control])

    # Update robot movement based on manual or automated behavior
    if manual_control:
        # Manual robot control with keyboard for speed and rotation
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            speed = min(speed + 0.1, max_speed)
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            speed = max(speed - 0.1, -max_speed)
        else:
            speed *= 0.9  # Friction effect

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            angular_speed = max(angular_speed - 0.01, -max_angular_speed)
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            angular_speed = min(angular_speed + 0.01, max_angular_speed)
        else:
            angular_speed *= 0.8  # Rotational friction

        heading += angular_speed
        heading %= 2 * math.pi
        x += speed * math.cos(heading)
        y += speed * math.sin(heading)

    else:
        # Automated robot control:
        # Aggressor mode -> turn toward light; Fear mode -> turn away
        if is_aggressor:
            vl = sr * 2
            vr = sl * 2
        else:
            vl = sl * 2
            vr = sr * 2

        heading += C * (vr - vl)
        heading %= 2 * math.pi
        v = (vl + vr) / 2
        x += v * math.cos(heading)
        y += v * math.sin(heading)

    # Wrap robot around screen edges (toroidal space)
    x %= WIDTH
    y %= HEIGHT

    # --- Rendering ---
    win.fill((0, 0, 0))

    # Draw circular rings of light intensity background
    for r in range(LIGHT_RADIUS, 0, -5):
        intensity = max(0, 255 * (r / LIGHT_RADIUS))
        color = (int(intensity), int(intensity), 100)
        pygame.draw.circle(win, color, (int(LIGHT_POS[0]), int(LIGHT_POS[1])), r, width=3)

    # Draw light source and robot
    pygame.draw.circle(win, (255, 255, 100), (int(LIGHT_POS[0]), int(LIGHT_POS[1])), 10)
    pygame.draw.circle(win, (100, 200, 255), (int(x), int(y)), ROBOT_RADIUS)

    # Draw sensors
    pygame.draw.circle(win, (0, 255, 0), (int(sl_pos[0]), int(sl_pos[1])), 5)
    pygame.draw.circle(win, (255, 0, 0), (int(sr_pos[0]), int(sr_pos[1])), 5)

    # Display robot info and controls on screen
    info1 = f"Robot pos: ({x:.1f}, {y:.1f})"
    info2 = f"Theta (deg): {math.degrees(heading) % 360:.1f}"
    info3 = f"Sensor L light intensity: {sl:.3f}"
    info4 = f"Sensor R light intensity: {sr:.3f}"
    control_instructions = [
        "W/UP: Forward; S/DOWN: Backward; A/LEFT: Left; D/RIGHT: Right; Click: Move Light Source"
    ]
    mode_text = f"Mode: {'Aggressor' if is_aggressor else 'Fear'}"
    texts = [info1, info2, info3, info4] + control_instructions + [mode_text]
    for i, text in enumerate(texts):
        surf = font.render(text, True, (255, 255, 255))
        win.blit(surf, (10, 10 + i * 20))

    # Draw toggle mode button
    pygame.draw.rect(win, (50, 50, 50), button_rect)
    btn_text = font.render(mode_text, True, (255, 255, 255))
    win.blit(btn_text, (button_rect.x + 10, button_rect.y + 5))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()

# --- Plot 1: Light Intensity Field ---
plt.style.use('seaborn-v0_8-whitegrid')

plt.figure(figsize=(8, 6))
ax1 = plt.gca()
ax1.set_aspect("equal", adjustable="box")
ax1.axis("equal")

xv, yv = np.meshgrid(np.linspace(0, WIDTH, WIDTH), np.linspace(0, HEIGHT, HEIGHT))
positions = np.dstack((xv, yv))

def light_intensity_np(pos):
    dx = pos[..., 0] - LIGHT_POS[0]
    dy = pos[..., 1] - LIGHT_POS[1]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    return np.maximum(0, 1 - dist / LIGHT_RADIUS)

field = light_intensity_np(positions)

# Intensity map using 'inferno' colormap
plt.imshow(field, extent=(0, WIDTH, 0, HEIGHT), origin='lower', cmap='inferno', alpha=0.8)
plt.scatter(*LIGHT_POS, color='cyan', edgecolor='k', label='Light Source')

plt.title("Light Intensity Field")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.gca().invert_yaxis()
plt.legend(framealpha=1, facecolor='white', edgecolor='black')
plt.tight_layout()
plt.show()

# --- Plot 2: Robot Trajectory ---

plt.figure(figsize=(8, 6))
ax2 = plt.gca()
ax2.set_aspect("equal", adjustable="box")
ax2.axis("equal")

robot_traj = np.array(robot_traj)

# Boolean masks for trajectory sections by mode and control
agg_auto = (robot_traj[:, 2] == True) & (robot_traj[:, 3] == False)
agg_manual = (robot_traj[:, 2] == True) & (robot_traj[:, 3] == True)
fear_auto = (robot_traj[:, 2] == False) & (robot_traj[:, 3] == False)
fear_manual = (robot_traj[:, 2] == False) & (robot_traj[:, 3] == True)

# Helper to plot multi-part trajectories split by screen edge wrap
def plot_segments(xy_segments, color, linestyle, label):
    for i, segment in enumerate(xy_segments):
        plt.plot(segment[:, 0], segment[:, 1], color=color, linestyle=linestyle,
                 label=label if i == 0 else None)  # Label only first segment

# Split trajectories to avoid lines crossing screen edges
agg_auto_segs = split_trajectory_on_wrap(robot_traj[agg_auto, 0:2], WIDTH, HEIGHT)
agg_manual_segs = split_trajectory_on_wrap(robot_traj[agg_manual, 0:2], WIDTH, HEIGHT)
fear_auto_segs = split_trajectory_on_wrap(robot_traj[fear_auto, 0:2], WIDTH, HEIGHT)
fear_manual_segs = split_trajectory_on_wrap(robot_traj[fear_manual, 0:2], WIDTH, HEIGHT)

# Plot all segments with appropriate styles
plot_segments(agg_auto_segs, 'skyblue', '-', 'Aggressor')
plot_segments(agg_manual_segs, 'skyblue', ':', 'Aggressor (User Controlled)')
plot_segments(fear_auto_segs, 'orange', '-', 'Fear')
plot_segments(fear_manual_segs, 'orange', ':', 'Fear (User Controlled)')

plt.scatter(*LIGHT_POS, color='cyan', edgecolor='k', label='Light Source')

# Draw yellow dotted circle for light radius
light_circle = patches.Circle(LIGHT_POS, LIGHT_RADIUS, edgecolor='red', facecolor='none',
                              linestyle=':', linewidth=2)
ax2.add_patch(light_circle)

# Add patch legend entry for the light radius circle
light_radius_patch = mpatches.Patch(edgecolor='red', facecolor='none', linestyle=':', linewidth=2,
                                   label='Light Radius')
handles, labels = ax2.get_legend_handles_labels()
handles.append(light_radius_patch)
labels.append(light_radius_patch.get_label())
plt.legend(handles=handles, labels=labels, framealpha=1, facecolor='white', edgecolor='black')

plt.title("Robot Trajectory (segmented by mode and control)")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
