from bayes_opt import BayesianOptimization
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
#from PyFoam.Infrastructure.ClusterJob import SolverRunner
# from PyFoam.Execution.UtilityRunner import UtilityRunner
# from PyFoam.Execution.AnalyzedRunner import AnalyzedRunner
# from PyFoam.Execution.BasicRunner import BasicRunner
#from PyFoam.Analysis.CompactAnalyzer import CompactAnalyzer
from datetime import datetime
import shutil
import os
from os import path
import numpy as np
import subprocess
import argparse

class OpenFOAMOptimizer:
    def __init__(self, base_case="mefenemic-base", n_parallel=8, serial=False):
        self.base_case = base_case
        self.n_parallel = n_parallel
        self.serial=serial

    def update_u_conditions(self, case_dir, amplitude, frequency):
        """Update the parameters in u-conditions.py"""
        u_conditions_path = os.path.join(case_dir, "u-conditions.py")
        
        # Read the file content
        with open(u_conditions_path, 'r') as file:
            lines = file.readlines()
            
        # Update the parameters
        for i, line in enumerate(lines):
            if 'freq = ' in line:
                lines[i] = f'freq = {frequency}  # Hz\n'
            elif 'amp = ' in line:
                lines[i] = f'amp = {amplitude:.6f}  # m\n'
        
        # Write back the updated content
        with open(u_conditions_path, 'w') as file:
            file.writelines(lines)
        
    def setup_case(self, amplitude, frequency):
        """Set up and run a single OpenFOAM case with given parameters"""
        # Create unique case directory
        identifier = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        case = identifier
        
        # Copy base case
        print(f'Creating case directory: {case}')
        shutil.copytree(self.base_case, case)
        
        self.update_u_conditions(case, amplitude, frequency)

        try:
            original_dir = os.getcwd()
            os.chdir(case)
            subprocess.run(["python3", "u-conditions.py"], check=True)
            os.chdir(original_dir)
        except subprocess.CalledProcessError as e:
            print(f"Error executing u-conditions.py: {e}")
            return None

        if not self.serial:   
            # Decompose domain
            decomposer = subprocess.run(
                ['decomposePar', '-case', case],
                capture_output=True,
                text=True
            )
            if decomposer.returncode != 0:
                    print("Domain decomposition failed")
                    return None
            
            # Run parallel simulation
            sim_command = f"mpirun -np {self.n_parallel} pimpleScalarsFoam -parallel -case {case}"
            simulation = subprocess.run(
                sim_command.split(),
                capture_output=True,
                text=True
            )

            if simulation.returncode != 0:
                print("Simulation failed")
                return None
                    
            # Reconstruct the solution
            reconstructor = subprocess.run(
                ['reconstructPar', '-case', case, '-time', '3'],
                capture_output=True,
                text=True
            )
        
        else:
            # Serial execution
            simulation = subprocess.run(
                ['pimpleScalarsFoam', '-case', case],
                capture_output=True,
                text=True
            )
            
            if simulation.returncode != 0:
                print("Simulation failed")
                return None
        # Extract crystal size (implement your specific post-processing)
        max_crystal_size = self.analyze_results(case)
        
        return max_crystal_size
    
    def analyze_results(self, case_dir):
        try:
            # Change to case directory
            original_dir = os.getcwd()
            os.chdir(case_dir)
            
            # Run crystalsizes.py
            subprocess.run(["python3", "crystalsizes.py"], check=True)
            
            # Read the output to get max crystal size
            # We'll use numpy to load the data and get max value
            # The script already filters the data, so we can use the max directly
            with open("output.txt", "r") as f:
                max_size = float(f.read())
            
            # Change back to original directory
            os.chdir(original_dir)
            
            return max_size
        
        except Exception as e:
            print(f"Error in analyzing results: {e}")
            os.chdir(original_dir)
            return 1e6  # Return a large value if analysis fails
    
    def objective_function(self, amplitude, frequency):
        """Objective function for Bayesian optimization"""
        try:
            max_crystal_size = self.setup_case(amplitude, frequency)
            # We return negative because BayesianOptimization maximizes
            if max_crystal_size is None:
                return -1e6  # Penalty for failed cases     
            return -max_crystal_size
        except Exception as e:
            print(f"Error in simulation: {e}")
            # Return a penalty value if simulation fails
            return -1e6

def run_optimization(serial=False):
    # Initialize optimizer
    optimizer = OpenFOAMOptimizer(base_case="mefenemic-base", n_parallel=8,serial=serial)
    
    # Define parameter bounds
    pbounds = {
        'amplitude': (0.001, 0.008),  # Adjust these ranges based on your needs
        'frequency': (1.0, 8.0)
    }
    
    # Initialize Bayesian Optimization
    optimizer_bo = BayesianOptimization(
        f=lambda **params: optimizer.objective_function(**params),
        pbounds=pbounds,
        random_state=1,
    )
    
    # Add logging
    from bayes_opt.logger import JSONLogger
    from bayes_opt.event import Events
    logger = JSONLogger(path="./logs.json")
    optimizer_bo.subscribe(Events.OPTIMIZATION_STEP, logger)
    
    # Run optimization
    optimizer_bo.maximize(
        init_points=3,  # Number of initial random points
        n_iter=10,      # Number of optimization iterations
    )
    
    print("Best result:", optimizer_bo.max)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', action='store_true', 
                      help='Run in serial mode instead of parallel')
    args = parser.parse_args()
    
    run_optimization(serial=args.serial)
