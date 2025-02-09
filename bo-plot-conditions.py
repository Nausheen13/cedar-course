import json
import numpy as np
import matplotlib.pyplot as plt

# Read and parse the JSON data from file
data = []
with open('logs.json', 'r') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

# Filter out failed iterations (target = -1000)
filtered_data = [d for d in data if d['target'] != -1000000.0]

# Extract values
targets = [-d['target'] for d in filtered_data]  # Negating for minimization
amplitudes = [d['params']['amplitude'] for d in filtered_data]
frequencies = [d['params']['frequency'] for d in filtered_data]

# Create figure with two subplots
fig = plt.figure(figsize=(15, 6))

# Plot 1: Best Target Value Found So Far
ax1 = fig.add_subplot(121)
iterations = range(1, len(data) + 1)
best_so_far = np.minimum.accumulate(targets)
ax1.plot(iterations, targets, 'g-o', linewidth=2, markersize=8)
ax1.plot(iterations, best_so_far, 'r:', linewidth=2, label='Best value')
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('max.crystal size (m)', fontsize=12)
ax1.set_title('Saturation plot', fontsize=14, pad=10)
ax1.grid(True, alpha=0.3)

# Plot 2: 2D Scatter Plot
ax2 = fig.add_subplot(122, projection='3d')
scatter = ax2.scatter(amplitudes, frequencies, 
                     c=targets,  # Color by target value
                     cmap='viridis',
                     s=100,  # Size of points
                     alpha=0.6)

ax2.set_xlabel('Amplitude (m)', fontsize=12)
ax2.set_ylabel('Frequency (Hz)', fontsize=12)
ax2.set_title('Reactor oscillatory conditions exploration', fontsize=14, pad=10)
ax2.grid(True, alpha=0.3)

# Add colorbar
plt.colorbar(scatter, ax=ax2, label='max.crystal size (m)')
plt.tight_layout()

# Display plots
plt.savefig('Conditions-Optimisation.png')
plt.show()