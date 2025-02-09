from ntpath import join
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import interpolate
import shutil 
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from datetime import datetime
import math
import uuid
import subprocess
import pyDOE
from pyDOE import lhs
import csv
from datetime import datetime
from bayes_opt import BayesianOptimization
import argparse

def sin_line(x, b, a, c,off):
    return b + (a * np.sin(c * (x-off)))

def smooth(x,y,n_add):
    k = 5
    x_p = [x[n_add-k],x[n_add],x[n_add+k]]
    y_mid_new = (((y[n_add-k] + y[n_add+k])/2) + y[n_add])/2
    y_p = [y[n_add-k],y_mid_new,y[n_add+k]]
    #print(y_mid_new)
    x_new = np.linspace(x[n_add-k],x[n_add+k],2*k)
    y_new = interp1d(x_p,y_p,kind='quadratic')(x_new)

    x[n_add-k:n_add+k] = x_new
    y[n_add-k:n_add+k] = y_new
    return x,y

def build_arrays(p1, p2, p3):
    x = np.linspace(0, 3, 180)
    # p1 = 0.25 # 0.1 <-> 0.4
    # p2 = 5    # 3 <-> 6

    if p1 > 0.70:
        print("--- You have entered an invalid geometry ---")
        print("The value of p1 is too high (", p1, "> 0.70 )")
        return

    if p1 < 0.1:
        print("--- You have entered an invalid geometry ---")
        print("The value of p1 is too low (", p1, "< 0.1 )")
        return

    if p2 < 3:
        print("--- You have entered an invalid geometry ---")
        print("The value of p2 is too low (", p2, "< 3 )")
        return

    if p2 > 6:
        print("--- You have entered an invalid geometry ---")
        print("The value of p2 is too high (", p2, "> 6 )")
        return
    
    if p3 < 0:
        print("--- You have entered an invalid geometry ---")
        print("The value of p3 is too low (", p3, "< 0 )")
        return 

    if p3 > np.pi/2:
        print("--- You have entered an invalid geometry ---")
        print("The value of p3 is too high (", p3, "> pi/2 )")
        return 

    y1 = sin_line(x, 0.5, p1, p2,p3)
    y2 = sin_line(x, 0.0, 0.25, p2,p3)

    x1 = x 
    x2 = x
    n_add = 50
    add_start_x = list(np.linspace(x1[0]-1.0,x1[0],n_add,endpoint=False))
    add_end_x = list(np.flip(np.linspace(x1[-1]+1.0,x1[-1],n_add,endpoint=False)))

    y1_B= y1[0]
    y1_A= y2[0]+0.5
    add_start_y1= [0]*len(add_start_x)
    add_start_y1[0]= y1_A; add_start_y1[-1]=y1_B
    f_y1_start = interpolate.interp1d([add_start_x[0], add_start_x[-1]], [add_start_y1[0], add_start_y1[-1]], kind='linear')

    y1_C= y1[-1]
    y1_D= y2[-1]+0.5
    add_end_y1 = [0] * len(add_end_x)
    add_end_y1[0]= y1_C; add_end_y1[-5:]=[y1_D]*5
 
    f_y1_end = interpolate.interp1d([add_end_x[0], add_end_x[-1]], [add_end_y1[0], add_end_y1[-1]], kind='linear')
    #y1 = np.append(np.append([y1[0] for i in range(n_add)],y1),[y1[-1]for i in range(n_add)])#edit this
    
    y1 = np.append(np.append(f_y1_start(add_start_x),y1),f_y1_end(add_end_x))
    y2 = np.append(np.append([y2[0] for i in range(n_add)],y2),[y2[-1] for i in range(n_add)])

    end= len(y1)
    print('y1 start1 and y1 end1',y1[0],y2[0])
    print('width',(y1[0]+abs(y2[0])))

    x1 = np.append(np.append(add_start_x,x1),add_end_x)
    x2 = np.append(np.append(add_start_x,x2),add_end_x)

    x1,y1 = smooth(x1,y1,n_add)
    x2,y2 = smooth(x2,y2,n_add)
    x1,y1 = smooth(x1,y1,n_add+180)
    x2,y2 = smooth(x2,y2,n_add+180)

    l11 = [
        """(\t""" + str(x1[i]) + "\t" + str(y1[i]) + """\t0\t)"""
        for i in range(len(x1))
    ]
    l12 = [
        """(\t""" + str(x1[i]) + "\t" + str(y1[i]) + """\t0.1\t)"""
        for i in range(len(x1))
    ]

    l21 = [
        """(\t""" + str(x2[i]) + "\t" + str(y2[i]) + """\t0\t)"""
        for i in range(len(x2))
    ]
    l22 = [
        """(\t""" + str(x2[i]) + "\t" + str(y2[i]) + """\t0.1\t)"""
        for i in range(len(x2))
    ]

    return l11, l12, l21, l22,x1,y1,x2,y2

def cleanup_and_move_files(path, main_folder, sample_number):
    """
    Moves reactor_geometry.png and slicePlaneXY.vtp files to the main folder,
    renames them with the sample number, and deletes the case folder.
    """


def build_mesh(p1, p2,p3,path, main_folder,sample_number):
    shutil.copytree("mefenemic-base", path)
    l11, l12, l21, l22,x1,y1,x2,y2 = build_arrays(p1, p2,p3)

    plt.figure()
    #plt.scatter(x1,y1)
    plt.plot(x1, y1, c="k")
    plt.plot(x2, y2, c="k")
    plt.xlim(min(x1)-0.1,max(x2)+0.1)
    #plt.xlim(2.5,3.5)
    plt.ylim(-0.5-2,max(x2)-2)
    plt.grid()
    plt.title(f'Reactor Geometry - p1={p1:.3f}, p2={p2:.3f}, p3={p3:.3f}')
    plt.savefig(path+"/reactor_geometry.png")

    with open(path+"/system/blockMeshDict", "rb") as f:
        lines = f.readlines()

    l = 0
    s = []
    s.append(str(lines[l]).split("""b'""")[-1].split("""\\n""")[0])
    while "polyLine" not in str(lines[l]):
        s.append(str(lines[l]).split("""b'""")[-1].split("""\\n""")[0])
        l += 1

    i = 0 
    c = 0 
    while i < len(s):
        if '$' in s[i] and c in [0,4,7,3]:
            s[i] = s[i].replace('$xmin',str(x2[0]))
            s[i] = s[i].replace('$xmax',str(x1[0]))
            s[i] = s[i].replace('$ymax',str(y1[0]))
            s[i] = s[i].replace('$ymin',str(y2[0]))
            c += 1
            i += 1
        elif '$' in s[i] and c in [2,6,5,1]:
            s[i] = s[i].replace('$xmin',str(x1[-1]))
            s[i] = s[i].replace('$xmax',str(x2[-1]))
            s[i] = s[i].replace('$ymax',str(y1[-1]))
            s[i] = s[i].replace('$ymin',str(y2[-1]))
            c += 1
            i += 1 
        else:
            i += 1


    nums = ["0 1", "4 5", "3 2", "7 6"]
    lines_add = [l21, l22, l11, l12]

    for i in range(len(nums)):
        new_poly = "	polyLine " + nums[i] + " (" + lines_add[i][0]
        s.append(new_poly)
        for j in range(1, len(lines_add[i])):
            s.append(lines_add[i][j])
        s.append(""")""")
    s.append(""");""")

    while "boundary" not in str(lines[l]):
        l += 1

    for i in range(l, len(lines)):
        s.append(str(lines[l]).split("""b'""")[-1].split("""\\n""")[0])
        l += 1


    with open(path+"/system/blockMeshDict", "w") as f:
        for item in s:
            f.write("%s\n" % item)
      
    subprocess.run(['blockMesh', '-case', path], check=True)
    #os.system('blockMesh -case '+path)
    #check_mesh_output= os.system('blockMesh -case '+path)
    check_mesh_process = subprocess.Popen(['checkMesh', '-case', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    check_mesh_output, check_mesh_error = check_mesh_process.communicate()

    if 'Failed' in check_mesh_output or 'Failed' in check_mesh_error:
        print(f"Mesh generation failed for {path}. Deleting folder.")
        shutil.rmtree(path)
        return None, "Mesh generation failed"
    
    #Extract mesh size information from the checkMesh output
    mesh_size_line = next((line for line in check_mesh_output.split('\n') if 'cells:' in line), None)
    mesh_size = int(mesh_size_line.split()[1]) if mesh_size_line else None

    return mesh_size

class GeometryOptimizer:
    def __init__(self, base_case="mefenemic-base", n_parallel=8, serial=True):
        self.base_case = base_case
        self.n_parallel = n_parallel
        self.serial= serial
        
    def setup_case(self, p1, p2, p3):
        """Set up and run a single case with given geometry parameters"""
        # Create unique case directory
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        unique_id = uuid.uuid4().hex[:8]
        case_path = f"case_{timestamp}_{unique_id}"
        
        try:
            # Copy base case and generate mesh
            print(f'Creating case directory: {case_path}')
            mesh_size = build_mesh(p1, p2, p3, case_path, '.', 0)
            
            if mesh_size is None:
                print("Mesh generation failed")
                # if os.path.exists(case_path):
                #     #shutil.rmtree(case_path)
                # return None

            if not self.serial:             
                # Run simulation with parallel decomposition
                decomposer = subprocess.run(
                    ['decomposePar', '-case', case_path],
                    capture_output=True,
                    text=True
                )
                
                if decomposer.returncode != 0:
                    print("Domain decomposition failed")
                    return None
                    
                # Run parallel simulation
                sim_command = f"mpirun -np {self.n_parallel} pimpleScalarsFoam -parallel -case {case_path}"
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
                    ['reconstructPar', '-case', case_path, '-time', '3'],
                    capture_output=True,
                    text=True
                )
            
            else:
                # Serial execution
                simulation = subprocess.run(
                    ['pimpleScalarsFoam', '-case', case_path],
                    capture_output=True,
                    text=True
                )
                
                if simulation.returncode != 0:
                    print("Simulation failed")
                    return None
            # Extract crystal size
            max_crystal_size = self.analyze_results(case_path)
            
            # Clean up case directory
            #shutil.rmtree(case_path)
            
            return max_crystal_size
            
        except Exception as e:
            print(f"Error in case setup: {e}")
            # if os.path.exists(case_path):
            #     shutil.rmtree(case_path)
            # return None
    
    def analyze_results(self, case_dir):
        try:
            # Change to case directory
            original_dir = os.getcwd()
            os.chdir(case_dir)
            
            # Run crystal size analysis
            subprocess.run(["python3", "crystalsizes.py"], check=True)
            
            # Read the maximum crystal size
            with open("output.txt", "r") as f:
                max_size = float(f.read())
            
            os.chdir(original_dir)
            return max_size
            
        except Exception as e:
            print(f"Error in analyzing results: {e}")
            os.chdir(original_dir)
            return 1e6  # Return large penalty value
    
    def objective_function(self, p1, p2, p3):
        """Objective function for Bayesian optimization"""
        try:
            max_crystal_size = self.setup_case(p1, p2, p3)
            if max_crystal_size is None:
                return -1e6  # Penalty for failed cases
            
            # Return negative because BayesianOptimization maximizes
            return -max_crystal_size
            
        except Exception as e:
            print(f"Error in simulation: {e}")
            return -1e6
    

def run_optimization(serial=True):
    # Initialize optimizer
    optimizer = GeometryOptimizer(base_case="mefenemic-base", n_parallel=8, serial=serial) #n_parallel is only use when serial=False
    
    # Define parameter bounds
    pbounds = {
        'p1': (0.1, 0.5),    # Amplitude parameter
        'p2': (3.0, 6.0),    # Frequency parameter
        'p3': (0.0, np.pi/2) # Phase offset parameter
    }
    
    # Initialize Bayesian Optimization
    optimizer_bo = BayesianOptimization(
        f=optimizer.objective_function,
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
