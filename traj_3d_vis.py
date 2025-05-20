import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Folder containing trajectory CSVs
folder = "trajectories"

# Set up 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each trajectory
for filename in os.listdir(folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(folder, filename)
        df = pd.read_csv(filepath)
        
        # Extract position columns
        x = df["position_x"]
        y = df["position_y"]
        z = df["position_z"]
        
        # Plot with label (without file extension)
        label = os.path.splitext(filename)[0]
        label = 'Trajectory ' + label
        ax.plot(x, y, z, label=label)

# Label axes
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title("3D Mosquito Trajectories")

# Optional: show legend
ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize='small')

plt.tight_layout()
plt.show()
