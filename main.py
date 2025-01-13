from bayes_opt import BayesianOptimization
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
#from PyFoam.Infrastructure.ClusterJob import SolverRunner
from PyFoam.Execution.UtilityRunner import UtilityRunner
from PyFoam.Execution.AnalyzedRunner import AnalyzedRunner
from PyFoam.Execution.BasicRunner import BasicRunner
#from PyFoam.Analysis.CompactAnalyzer import CompactAnalyzer
from datetime import datetime
import shutil
import os
from os import path
import numpy as np
import subprocess

class OpenFOAMOptimizer:
    def __init__(self, base_case="serpentine", n_parallel=16):
        self.base_case = base_case
        self.n_parallel = n_parallel
        
    def setup_case(self, amplitude, frequency):
        """Set up and run a single OpenFOAM case with given parameters"""
        # Create unique case directory
        identifier = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        case = identifier
        
        # Copy base case
        print(f'Creating case directory: {case}')
        shutil.copytree(self.base_case, case)
        
        # Modify boundary conditions
        velBC = ParsedParameterFile(path.join(case, "0", "U"))
        velBC["boundaryField"]["inlet"]["variables"][0] = f'"amp= {amplitude:.5f};"'
        velBC["boundaryField"]["inlet"]["variables"][1] = f'"freq= {frequency:.5f};"'
        velBC.writeFile()
        
        # Decompose domain
        decomposer = UtilityRunner(
            argv=["decomposePar", "-case", case],
            logname="decomposePar",
        )
        decomposer.start()
        
        # Run simulation
        run_command = f"mpirun -np {self.n_parallel} pimpleScalarsFoam -parallel"
        run = BasicRunner(
            argv=[run_command, "-case", case],
            logname="Solution",
        )
        run.start()

        reconstructor = UtilityRunner(
            argv=["reconstructPar", "-case", case, "-time", "3"],
            logname="reconstructPar",
        )
        reconstructor.start()
        
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
            return -max_crystal_size
        except Exception as e:
            print(f"Error in simulation: {e}")
            # Return a penalty value if simulation fails
            return -1e6

def run_optimization():
    # Initialize optimizer
    optimizer = OpenFOAMOptimizer(base_case="base", n_parallel=16)
    
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
        init_points=5,  # Number of initial random points
        n_iter=20,      # Number of optimization iterations
    )
    
    print("Best result:", optimizer_bo.max)

if __name__ == "__main__":
    run_optimization()