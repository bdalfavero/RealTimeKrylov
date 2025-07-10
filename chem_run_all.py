import json
import subprocess
from pathos.pools import ProcessPool

chi_tdvp = [30, 35, 40, 45, 50, 55, 60]
input_files = []
for i, chi in enumerate(chi_tdvp):
    with open("data/hf_input.json", "r") as f:
        input_dict = json.load(f)
    input_dict["chi_tdvp"] = chi
    input_filename = f"data/hf_input_{i}.json"
    with open(input_filename, "w") as f:
        json.dump(input_dict, f)
    input_files.append(input_filename)

def run_calc(input_file):
    output_file = input_file.replace("input", "output")
    subprocess.call(["python", "chem_krylov.py", input_file, output_file])

pool = ProcessPool(nodes=6)
pool.map(run_calc, input_files)