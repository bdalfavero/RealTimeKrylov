import json
import subprocess

chi = [10, 20, 30]
for i, chi_dmrg in enumerate(chi):
    input_dict = {
        "molec_name": "hydrogen",
        "chi_dmrg": chi_dmrg,
        "chi_tdvp": chi_dmrg,
        "T": 0.1,
        "dt": 0.01
    }
    input_file = f"data/hydrogen/input_{i}.json"
    states_file = f"data/hydrogen/states_{i}.pkl"
    energies_file = f"data/hydrogen/energies_{i}.hdf5"
    with open(input_file, "w") as f:
        json.dump(input_dict, f)
    
    subprocess.run(["python", "molec_gen_state.py", input_file, states_file])
    subprocess.run(["python", "molec_energies_from_states.py", states_file, energies_file])