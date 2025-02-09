import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read and parse the JSON data from file
data = []
with open('logs.json', 'r') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

# Filter out failed iterations (target = -1000)
filtered_data = [d for d in data if d['target'] != -1000000.0]

# Extract values from filtered data
targets = [-d['target'] for d in filtered_data]  
p1_values = [d['params']['p1'] for d in filtered_data]
p2_values = [d['params']['p2'] for d in filtered_data]
p3_values = [d['params']['p3'] for d in filtered_data]

# Create figure with two subplots
fig = plt.figure(figsize=(15, 6))

# Plot 1: Best Target Value Found So Far
ax1 = fig.add_subplot(121)
iterations = range(1, len(filtered_data) + 1)
best_so_far = np.minimum.accumulate(targets)
ax1.plot(iterations, targets, 'g-o', linewidth=2, markersize=8)
ax1.plot(iterations, best_so_far, 'r:', linewidth=2, label='Best value')
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('max.crystal size (m)', fontsize=12)
ax1.set_title('Saturation plot', fontsize=14, pad=10)
ax1.grid(True, alpha=0.3)

# Plot 2: 3D Parameter Space Exploration
ax2 = fig.add_subplot(122, projection='3d')
scatter = ax2.scatter(p1_values, p2_values, p3_values, c=targets, 
                     cmap='viridis', s=100, alpha=0.6)
ax2.set_xlabel('p1', fontsize=12)
ax2.set_ylabel('p2', fontsize=12)
ax2.set_zlabel('p3', fontsize=12)
ax2.set_title('Reactor design exploration', fontsize=14, pad=10)
colorbar = plt.colorbar(scatter, ax=ax2)
colorbar.set_label('max.crystal size (m)', fontsize=12)

# Adjust layout
plt.tight_layout()

# Display plots
plt.savefig('Geometry-Optimisation.png')
plt.show()
