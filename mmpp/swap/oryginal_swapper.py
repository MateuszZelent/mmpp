import os
import subprocess
import os
import numpy as np

def replace_variables_in_template(file_path, variables):
    with open(f"{main_path}{file_path}", 'r') as file:
        content = file.read()
    for key, value in variables.items():
        content = content.replace(f'{{{key}}}', str(value))
    return content

def raw_code(*args, **kwargs):
    raw_code.__globals__.update(kwargs)
    modified_template = replace_variables_in_template("template.mx3", kwargs)
    return modified_template

import os
import re

def znajdz_plik_mx3(numer_symulacji, katalog_glowny):
    """
    Przeszukuje katalog_glowny i jego podkatalogi, aby znaleźć plik .mx3 z numerem symulacji.
    """
    wzor_pliku = f"sim_{numer_symulacji}.*\\.mx3"
    for root, dirs, files in os.walk(katalog_glowny):
        for file in files:
            if re.search(wzor_pliku, file):
                return os.path.join(root, file)
    return None

def parsuj_parametry(sciezka):
    """
    Parsuje ścieżkę, aby wyodrębnić parametry materiałowe: dind, msat, aex, ku.
    """
    parametry = {}
    # Poprawione wyrażenie regularne do obsługi notacji naukowej
    wzor_parametrow = r"(?:sim_)?([a-z]+)[_:]([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    wyniki = re.findall(wzor_parametrow, sciezka)
    for parametr, wartosc in wyniki:
        try:
            parametry[parametr] = [float(wartosc)]
        except ValueError:
            continue  # Ignoruje nierozpoznane formaty
    return parametry




def gen_sbatch_script(name, path,main_path=None,destination_path=None):
    mx3_file = f"{path}.mx3"
    lock_file = f"{path}.mx3_status.lock"
    done_file = f"{path}.mx3_status.done"
    interrupted_file = f"{path}.mx3_status.interrupted"
    origin_path = f"{path}.zarr"
    file_name="/".join(path.split(main_path)[:-1])
    d_path = f"{destination_path}/{file_name}.zarr"
    
    relative_path = path.replace(main_path, "")
    final_path = destination_path + relative_path +".zarr"
    
    script = f"""#!/bin/bash -l
#SBATCH --job-name="{name}"
#SBATCH --mail-type=NONE
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=proxima,tesla
#SBATCH --gres=gpu:1

source /mnt/storage_3/home/kkingstoun/.bashrc
export TMPDIR="/mnt/storage_2/scratch/grant_398/mumax_kernels/"

# Define the original file path
mx3_file="{mx3_file}"

# Generate the new filename using the Python script
lock_file=$(python /mnt/storage_3/home/kkingstoun/scripts/generate_machine_key.py "$mx3_file")

# Change the file name to indicate the simulation is running
mv "$mx3_file" "$lock_file"

# Start continuous sync with rclone in the background
sync_folder="${{mx3_file%.mx3}}.zarr"
echo $sync_folder

# sync_command="/usr/bin/rclone sync $sync_folder zfn:{final_path} --progress"
# echo $sync_command

# Use a while loop for continuous sync until the simulation ends
# sync_function() {{
#     while true; do
#         eval $sync_command
#         sleep 10 # Adjust the sleep time as needed
#     done
# }}
# Start continuous sync in the background
# sync_function &

sync_pid=$!


# Function to clean up background process
cleanup() {{
    if kill -0 $sync_pid 2>/dev/null; then
        kill $sync_pid
        wait $sync_pid 2>/dev/null
    else
        echo "Process $sync_pid not found."
    fi
}}

# Trap SIGINT and SIGTERM to ensure cleanup
trap cleanup SIGINT SIGTERM EXIT


# Run the simulation and capture the exit status
/mnt/storage_3/home/kkingstoun/software/amumax -magnets=0 -o "$sync_folder" "$lock_file"
simulation_exit_status=$?

# Check if the simulation finished successfully
if [ $simulation_exit_status -eq 0 ]; then
    echo "FINISHED"
    /mnt/storage_3/home/kkingstoun/new_home/software/miniconda/bin/python3 /mnt/storage_2/scratch/pl0095-01/zelent/mannga/FMR/coupled_resonator/post_hybrid2.py "$sync_folder"
    eval $sync_command
    mv "$lock_file" "{done_file}"
else
    echo "INTERRUPTED"
    mv "$lock_file" "{interrupted_file}"
fi
echo "ALL JOBS DONE"
# Stop the background sync process
cleanup
"""
    return script

import os
import errno
import os
import numpy as np
import itertools


def create_path_if_not_exists(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            
def submit_python_code(code_to_execute,main_path=None,destination_path=None,cleanup=False,sbatch=True,check=False,force=False,full_name=False,last_param_name=None,*args, **kwargs):
    
    if len(kwargs) > 0 and last_param_name == None:
        last_param_name = list(kwargs.keys())[-1]
        
    sim_params = '_'.join([f"{key}_{format(val, '.5g') if isinstance(val, (int, float)) else val}" for key, val in kwargs.items() if key != last_param_name and key != "i" and key != "prefix" ])
    if main_path==None:    
        main_path = f"{os.getcwd()}/"
    
    par_sep = ","
    val_sep="_"
    path = f"{main_path}{kwargs['prefix']}/"+'/'.join([f"{key}{val_sep}{format(val, '.5g') if isinstance(val, (int, float)) else val}" for key, val in kwargs.items() if key != last_param_name and key != "i" and key != "prefix" ])+"/"
    if full_name==False:
        last_key, last_val = kwargs.popitem()
        sim_name = f"{last_key}{val_sep}{format(last_val, '.5g')}"
    else:
        sim_name_params = par_sep.join([f"{key}{val_sep}{format(val, '.5g')}" for key, val in kwargs.items() if key != "prefix" and key != "i"])
        sim_name = f"{sim_name_params}"
    
    create_path_if_not_exists(path)
    
    if os.path.isdir(f"{path}.zarr/end") and force!=True:
        print("Simulation already performed")
        return
    else:
        print(f"Starting new silumation: {kwargs['i']} - {path}{sim_name}.mx3")
        with open(f"{path}{sim_name}.mx3", "w") as f:
            f.write(code_to_execute)    
        
        if sbatch==True:    
            sim_sbatch= '_'.join([f"{key}_{val}" for key, val in kwargs.items() if key == "i" ])
                
            sim_sbatch_path = f"{main_path}{kwargs['prefix']}/sbatch/{sim_sbatch}.sb"
            create_path_if_not_exists(sim_sbatch_path)
            
            with open(sim_sbatch_path, "w") as f:
                f.write(gen_sbatch_script(sim_name,path + sim_name,main_path,destination_path))  
                
            try:
                subprocess.call(f'sbatch < {sim_sbatch_path}', shell=True)
            finally:
                if cleanup:
                    os.remove(sim_sbatch_path)

def submit_all_simulations(params, last_param_name,minsim=0,maxsim=None,main_path=None,destination_path=None, prefix=None, sbatch=True, full_name=False,cleanup=False, check=False, force=False):
    """
    Submit all possible simulations based on the given parameters.

    Parameters:
    params (dict): A dictionary of parameter names and their ranges as numpy arrays.
    last_param_name (str): The name of the parameter that should be iterated over last.
    prefix (str): The prefix for the output directory.
    sbatch (bool): Whether to submit jobs using sbatch.
    cleanup (bool): Whether to delete the sbatch script after submission.
    check (bool): Whether to check if the simulation has already finished before submitting.
    force (bool): Whether to force resubmission of the simulation even if it has already finished.

    Returns:
    None
    """
    param_names = list(params.keys())
    
    product = itertools.product(*params.values())
    for i, values in enumerate(product):
        if i<minsim:
            continue
        if i==maxsim:
            break
        kwargs = {"prefix": prefix, "i": i}
        for name, value in zip(param_names, values):
            kwargs[name] = value
        submit_python_code(
            raw_code(**kwargs),
            main_path=main_path,
            destination_path=destination_path,
            last_param_name=last_param_name,
            sbatch=sbatch,
            cleanup=cleanup,
            full_name=full_name,
            check=check,
            force=force,
            **kwargs
        )
        
def calculate_cases(params):
    total_cases = 1  # Zaczynamy od wartości jednostkowej, ponieważ wykonujemy mnożenie
    for key in params:
        total_cases *= len(params[key])  # Mnożymy przez długość każdej listy parametrów
    return total_cases
    
# /mnt/storage_2/scratch/pl0095-01/zelent/mannga/FMR/single_resonator/v10_selected/Tx_1e-06/xsize_475/ysize_600/sq_parm_0/rotation_0/B0_0.025/b01_0.001/anetnna_0/sim_90,anetnna_0.zarr
# /mnt/storage_2/scratch/pl0095-01/zelent/mannga/FMR/single_resonator/v10_selected/Tx_1e-06/xsize_600/ysize_350/sq_parm_0.5/rotation_90/B0_0.025/b01_0.001/anetnna_0/sim_118,anetnna_0.zarr
params = {
    "Material":[3],
    "Tx":[6000e-9],
    # "Tx": np.linspace(1000e-9,6000e-9, 6),
    # "xsize": np.linspace(350,600, 3),
    # "ysize": np.linspace(350,600, 3),
    "xsize": [100.0,200.0],
    "ysize": [100.0,200.0],
    "sq_parm": [1.0],
    "rotation": [0],
    # "rotation": np.linspace(0.0, 90.0, 3),
    # "b01": [0.001],
    "anetnna": [0],
    "B0": np.linspace(0.005,0.05,10),
}
num_cases = calculate_cases(params)
print("Total number of cases:", num_cases)


destination_path="/mnt/Primary/zfn/mateuszz/mannga/jobs/mateuszz/squircle/FMR/coupled_resonator/"
main_path="/mnt/storage_2/scratch/pl0095-01/zelent/mannga/FMR/coupled_resonator/"
post_processing_script="/mnt/storage_2/scratch/pl0095-01/zelent/mannga/FMR/coupled_resonator/post_hybrid.py"
submit_all_simulations(params, 
                       main_path=main_path, 
                       minsim=0,
                       maxsim=None,
                       destination_path=destination_path, 
                       last_param_name="B0",
                       prefix="v11",
                       sbatch=1,
                       full_name=False)