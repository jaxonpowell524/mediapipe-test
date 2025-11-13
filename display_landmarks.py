#CODE GENERATED WITH CHATGPT

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv("landmark_csv/landmarks-IMG_7083 (1).csv")      # <------csv filename--------
frames = df["frame"].max() + 1
num_landmarks = 33

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

for ax in (ax1, ax2):
    ax.set_xlim(1, 0)
    ax.set_ylim(0, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

ax1.view_init(elev=-0, azim=0, roll=270)
ax2.view_init(elev=-90, azim=90, roll=180)

points1 = ax1.scatter([], [], [], c='lime', s=25)
points2 = ax2.scatter([], [], [], c='lime', s=25)

POSE_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 12),
    (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32)
]
lines1 = [ax1.plot([], [], [], c='black', lw=1)[0] for _ in POSE_CONNECTIONS]
lines2 = [ax2.plot([], [], [], c='black', lw=1)[0] for _ in POSE_CONNECTIONS]

running = True
def on_close(event):
    global running
    running = False

fig.canvas.mpl_connect("close_event", on_close)

while running:
    for frame_idx in range(frames):
        if not running:
            break
        row = df[df["frame"] == frame_idx]
        if row.empty:
            continue

        x = [row[f"x{i}"].values[0] for i in range(num_landmarks)]
        y = [row[f"y{i}"].values[0] for i in range(num_landmarks)]
        z = [row[f"z{i}"].values[0] for i in range(num_landmarks)]

        # update plots
        for pts, lines, ax in [(points1, lines1, ax1), (points2, lines2, ax2)]:
            pts._offsets3d = (x, y, z)
            for j, (a, b) in enumerate(POSE_CONNECTIONS):
                lines[j].set_data([x[a], x[b]], [y[a], y[b]])
                lines[j].set_3d_properties([z[a], z[b]])

        plt.draw()
        plt.pause(0.03)

plt.close(fig)
