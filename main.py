import pygame
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.patches as mpatches

# --- Initialization ---
pygame.init()
pygame.font.init()

infoObject = pygame.display.Info()
WIDTH, HEIGHT = infoObject.current_w, infoObject.current_h

FPS = 60
LIGHT_POS = [WIDTH // 2, HEIGHT // 2]
LIGHT_RADIUS = 400

ROBOT_RADIUS = 15
SENSOR_OFFSET_ANGLE = math.pi / 4
SENSOR_OFFSET_DIST = 20
C = 0.05

win = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont('Arial', 18)

x, y = 0, 0
heading = 3 * math.pi / 2
speed = 0
angular_speed = 0
max_speed = 3
max_angular_speed = 0.1

is_aggressor = True

button_width, button_height = 220, 30
button_rect = pygame.Rect(WIDTH - button_width - 10, 10, button_width, button_height)

def light_intensity(pos):
    dx = pos[0] - LIGHT_POS[0]
    dy = pos[1] - LIGHT_POS[1]
    dist = math.sqrt(dx * dx + dy * dy)
    intensity = max(0, 1 - dist / LIGHT_RADIUS)
    return intensity

def sensor_position(x, y, heading, offset_angle):
    angle = heading + offset_angle
    sx = x + SENSOR_OFFSET_DIST * math.cos(angle)
    sy = y + SENSOR_OFFSET_DIST * math.sin(angle)
    return sx % WIDTH, sy % HEIGHT

def split_trajectory_on_wrap(xy, width, height):
    if len(xy) == 0:
        return []
    segments = []
    segment = [xy[0]]
    for i in range(1, len(xy)):
        prev = xy[i-1]
        curr = xy[i]
        dx = abs(curr[0] - prev[0])
        dy = abs(curr[1] - prev[1])
        if dx > width / 2 or dy > height / 2:
            segments.append(np.array(segment))
            segment = [curr]
        else:
            segment.append(curr)
    if segment:
        segments.append(np.array(segment))
    return segments

robot_traj = []  # Logs: x, y, mode, manual control

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos):
                is_aggressor = not is_aggressor
            else:
                LIGHT_POS[0], LIGHT_POS[1] = event.pos

    keys = pygame.key.get_pressed()
    manual_control = any(keys)

    sl_pos = sensor_position(x, y, heading, SENSOR_OFFSET_ANGLE)
    sr_pos = sensor_position(x, y, heading, -SENSOR_OFFSET_ANGLE)
    sl = light_intensity(sl_pos)
    sr = light_intensity(sr_pos)
    robot_traj.append([x, y, is_aggressor, manual_control])

    if manual_control:
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            speed = min(speed + 0.1, max_speed)
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            speed = max(speed - 0.1, -max_speed)
        else:
            speed *= 0.9

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            angular_speed = max(angular_speed - 0.01, -max_angular_speed)
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            angular_speed = min(angular_speed + 0.01, max_angular_speed)
        else:
            angular_speed *= 0.8

        heading += angular_speed
        heading %= 2 * math.pi
        x += speed * math.cos(heading)
        y += speed * math.sin(heading)
    else:
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

    x %= WIDTH
    y %= HEIGHT

    win.fill((0, 0, 0))

    for r in range(LIGHT_RADIUS, 0, -5):
        intensity = max(0, 255 * (r / LIGHT_RADIUS))
        color = (int(intensity), int(intensity), 100)
        pygame.draw.circle(win, color, (int(LIGHT_POS[0]), int(LIGHT_POS[1])), r, width=3)

    pygame.draw.circle(win, (255, 255, 100), (int(LIGHT_POS[0]), int(LIGHT_POS[1])), 10)
    pygame.draw.circle(win, (100, 200, 255), (int(x), int(y)), ROBOT_RADIUS)

    pygame.draw.circle(win, (0, 255, 0), (int(sl_pos[0]), int(sl_pos[1])), 5)
    pygame.draw.circle(win, (255, 0, 0), (int(sr_pos[0]), int(sr_pos[1])), 5)

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

    pygame.draw.rect(win, (50, 50, 50), button_rect)
    btn_text = font.render(mode_text, True, (255, 255, 255))
    win.blit(btn_text, (button_rect.x + 10, button_rect.y + 5))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()

## --- Light intensity field plot ---
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
plt.imshow(field, extent=(0, WIDTH, 0, HEIGHT), origin='lower', cmap='inferno', alpha=0.8)

plt.scatter(*LIGHT_POS, color='cyan', edgecolor='k', label='Light Source')

plt.title("Light Intensity Field")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.gca().invert_yaxis()
plt.legend(framealpha=1, facecolor='white', edgecolor='black')
plt.tight_layout()
plt.show()

## --- Robot trajectory plot ---

plt.figure(figsize=(8, 6))
ax2 = plt.gca()
ax2.set_aspect("equal", adjustable="box")
ax2.axis("equal")

robot_traj = np.array(robot_traj)
agg_auto = (robot_traj[:, 2] == True) & (robot_traj[:, 3] == False)
agg_manual = (robot_traj[:, 2] == True) & (robot_traj[:, 3] == True)
fear_auto = (robot_traj[:, 2] == False) & (robot_traj[:, 3] == False)
fear_manual = (robot_traj[:, 2] == False) & (robot_traj[:, 3] == True)

def plot_segments(xy_segments, color, linestyle, label):
    for i, segment in enumerate(xy_segments):
        plt.plot(segment[:, 0], segment[:, 1], color=color, linestyle=linestyle, label=label if i == 0 else None)

agg_auto_segs = split_trajectory_on_wrap(robot_traj[agg_auto, 0:2], WIDTH, HEIGHT)
agg_manual_segs = split_trajectory_on_wrap(robot_traj[agg_manual, 0:2], WIDTH, HEIGHT)
fear_auto_segs = split_trajectory_on_wrap(robot_traj[fear_auto, 0:2], WIDTH, HEIGHT)
fear_manual_segs = split_trajectory_on_wrap(robot_traj[fear_manual, 0:2], WIDTH, HEIGHT)

plot_segments(agg_auto_segs, color='skyblue', linestyle='-', label='Aggressor')
plot_segments(agg_manual_segs, color='skyblue', linestyle=':', label='Aggressor (User Controlled)')
plot_segments(fear_auto_segs, color='orange', linestyle='-', label='Fear')
plot_segments(fear_manual_segs, color='orange', linestyle=':', label='Fear (User Controlled)')

plt.scatter(*LIGHT_POS, color='cyan', edgecolor='k', label='Light Source')

light_circle = patches.Circle(LIGHT_POS, LIGHT_RADIUS, edgecolor='red', facecolor='none',
                              linestyle=':', linewidth=2)
ax2.add_patch(light_circle)

light_radius_patch = mpatches.Patch(edgecolor='red', facecolor='none', linestyle=':', linewidth=2, label='Light Radius')
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
