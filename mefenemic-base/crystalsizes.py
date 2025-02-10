import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def read_openfoam_field(filepath):
    """Read OpenFOAM scalar field file with header."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
        # Find where the data starts
        start_idx = 0
        for i, line in enumerate(lines):
            if 'internalField' in line:
                # Next line contains number of cells
                n_cells = int(lines[i+1])
                start_idx = i + 2
                break
        
        # Read the data values (skipping parentheses)
        data = []
        for line in lines[start_idx+1:]:  # Skip the opening parenthesis
            if ')' in line:  # Stop at closing parenthesis
                break
            try:
                value = float(line.strip())
                data.append(value)
            except ValueError:
                continue
                
        return np.array(data)

# Specify the time directory and field name
time = "3"  # Change this to your desired time
field_name = "crystalSize"
case_dir = "."  # Change this to your case directory path if needed

# Construct the file path
filepath = f"{case_dir}/{time}/{field_name}"

# Read the data
data = read_openfoam_field(filepath)

if len(data) > 0:
    # Find the minimum non-zero value
    non_zero_min = np.min(data[data > 0])
    print(f"Minimum non-zero value in data: {non_zero_min:.3e}")
    
    # Use this as threshold
    min_threshold = non_zero_min
    
    # Filter out values below threshold
    data_filtered = data[data > min_threshold]
    
    if len(data_filtered) > 0:
        # Create histogram
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        n, bins, patches = plt.hist(data_filtered, bins=10, density=True, 
                                  alpha=0.7, color='blue', edgecolor='black',
                                  label='Histogram')
        
        # Add a kernel density estimate if there's variation in the data
        if data_filtered.std() > 0:
            from scipy import stats
            kde = stats.gaussian_kde(data_filtered)
            x_range = np.linspace(data_filtered.min(), data_filtered.max(), 200)
            plt.plot(x_range, kde(x_range), 'r-', lw=2, 
                     label='Kernel Density Estimate')
        
        # Customize the plot
        plt.xlabel(f'{field_name} [m]')
        plt.ylabel('Frequency Density')
        plt.title(f'Distribution of {field_name} at t = {time}s\n(values ≥ {min_threshold:.3e})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add statistical information
        stats_text = f'Total cells: {len(data)}\n'
        stats_text += f'Cells ≥ threshold: {len(data_filtered)}\n'
        stats_text += f'Mean (filtered): {data_filtered.mean():.3e}\n'
        stats_text += f'Std (filtered): {data_filtered.std():.3e}\n'
        stats_text += f'Min (filtered): {data_filtered.min():.3e}\n'
        stats_text += f'Max: {data_filtered.max():.3e}'
        plt.text(0.95, 0.95, stats_text,
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'{field_name}_histogram_{time}_filtered.png', dpi=300, bbox_inches='tight')
        #plt.show()
        
        # Print some basic statistics
        print(f"\nStatistics for {field_name} at t = {time}:")
        print(f"Total number of cells: {len(data)}")
        print(f"Number of cells ≥ threshold: {len(data_filtered)}")
        print(f"Mean (filtered): {data_filtered.mean():.3e}")
        print(f"Standard deviation (filtered): {data_filtered.std():.3e}")
        print(f"Min (filtered): {data_filtered.min():.3e}")
        print(f"Max: {data_filtered.max():.3e}")

        max_size = data_filtered.max()  # Or any computed value
        with open("output.txt", "w") as f:
            f.write(str(max_size))
    else:
        print("No values above the threshold found in the data")
else:
    print(f"Could not read data from {filepath}")
