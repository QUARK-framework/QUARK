import os
import json
import pickle
import numpy as np
import yaml
from pathlib import Path



def fetch_data(file_path):
    """
    Loads data from .npy, .pkl, .json, or .yml/.yaml files.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {os.path.abspath(file_path)}")
        return None

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    try:
        if ext == '.npy':
            data = np.load(file_path, allow_pickle=True)

        elif ext == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        elif ext in ['.yml', '.yaml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

        else:
            print(f"Unsupported format: {ext}")
            return None

    except Exception as e:
        print(f"Failed to load: {e}")
        return None
    
    return data


def get_precission(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {os.path.abspath(file_path)}")
        return None

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    try:
        if ext == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            print(f"Unsupported format: {ext}")
            return None
    except Exception as e:
        print(f"Failed to load: {e}")
        return None
    
    precission = data.get('precision', None) if isinstance(data, dict) else None

    return precission


def get_probability_distribution(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {os.path.abspath(file_path)}")
        return None

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    try:
        if ext == '.npy':
            data = np.load(file_path, allow_pickle=True)
        else:
            print(f"Unsupported format: {ext}")
            return None
    except Exception as e:
        print(f"Failed to load: {e}")
        return None
    
    probability_distribution = np.asarray(data)

    return probability_distribution


def get_config_parameters(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {os.path.abspath(file_path)}")
        return None

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    try:
        if ext in ['.yml', '.yaml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        else:
            print(f"Unsupported format: {ext}")
            return None

    except Exception as e:
        print(f"Failed to load: {e}")
        return None

    try:
        application = data["application"]

        # Top level
        n_qubits = application["config"]["n_qubits"][0]

        # Discrete Data
        data_module = application["submodules"][0]
        train_size = data_module["config"]["train_size"][0]

        # CircuitCardinality
        circuit_module = data_module["submodules"][0]
        depth = circuit_module["config"]["depth"][0]

        # LibraryQiskit
        library_module = circuit_module["submodules"][0]
        backend = library_module["config"]["backend"][0]
        n_shots = library_module["config"]["n_shots"][0]

        # QGAN
        training_module = library_module["submodules"][0]
        training_config = training_module["config"]

        config_parameters = {
            "n_qubits": n_qubits,
            "depth": depth,
            "data": data_module["name"].lower(),
            "circuit": circuit_module["name"]
                .replace("Circuit", "")
                .lower(),
            "library": library_module["name"]
                .replace("Library", "")
                .lower(),
            "backend": backend,
            "n_shots": n_shots,
            "training": training_module["name"],
            "repetitions": data.get("repetitions"),
            "ML metrics": {
                "train_size": train_size,
                "batch_size": training_config["batch_size"][0],
                "device": training_config["device"][0],
                "epochs": training_config["epochs"][0],
                "learning_rate_discriminator":
                    training_config["learning_rate_discriminator"][0],
                "learning_rate_generator":
                    training_config["learning_rate_generator"][0],
                "loss": training_config["loss"][0],
                "pretrained": training_config["pretrained"][0],
            }
        }

    except Exception as e:
        print(f"Failed to parse config structure: {e}")
        return None

    return config_parameters


def get_runtimes(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {os.path.abspath(file_path)}")
        return None

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    try:
        if ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            print(f"Unsupported format: {ext}")
            return None

    except Exception as e:
        print(f"Failed to load: {e}")
        return None

    runtimes = {}

    def extract_module_name(module):
        """
        Extracts module name from module_src path.
        Example:
        src/.../discrete_data.py -> discrete_data
        """
        module_src = module.get("module_src", "")

        if module_src:
            return os.path.splitext(os.path.basename(module_src))[0]

        return "unknown_module"

    def extract_module_times(module):
        """
        Recursively extracts timing information from modules.
        """
        module_name = extract_module_name(module)

        metrics = module.get("metrics", {})

        runtimes[module_name] = {
            "total_time": metrics.get("total_time"),
            "total_time_unit": metrics.get("total_time_unit"),
            "preprocessing_time": metrics.get("preprocessing_time"),
            "preprocessing_time_unit": metrics.get("preprocessing_time_unit"),
            "postprocessing_time": metrics.get("postprocessing_time"),
            "postprocessing_time_unit": metrics.get("postprocessing_time_unit"),
        }

        # Recursively process submodules
        for submodule in module.get("submodules", []):
            extract_module_times(submodule)

    # Start recursion from application root
    if "application" in data:
        extract_module_times(data["application"])

    # Add overall runtime
    runtimes["summed_time"] = {
        "total_time": data.get("total_time"),
        "total_time_unit": data.get("total_time_unit")
    }

    return runtimes


############

def get_paths_to_data(config_path_name, noisy=False):
    # 1. Setup the base path
    if noisy:        
        base_path = Path(r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\sorted\noisy")
    else:   
        base_path = Path(r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\sorted\not_noisy")
    
    # 2. Construct the full path to the specific config folder
    target_dir = base_path / config_path_name
    
    if not target_dir.exists():
        print(f"Directory not found: {target_dir}")
        return []

    all_run_data = []

    # 3. Iterate through all folders starting with 'generativemodeling'
    for gen_folder in target_dir.glob("generativemodeling-*"):
        
        # We search recursively (rglob) within this specific run folder
        # to find the deeply nested files.
        run_files = {
            "run_folder": str(gen_folder),
            "config_yml": next(gen_folder.glob("config.yml"), None),
            "results_json": next(gen_folder.glob("results.json"), None),
            # Using * to handle the potential underscore in 'metrics_1'
            "metrics_pkl": next(gen_folder.rglob("record_gen_metrics*.pkl"), None),
            "histogram_npy": next(gen_folder.rglob("histogram_generated.npy"), None)
        }
        
        # Convert Path objects to strings for easier use later
        for key in run_files:
            if run_files[key] and key != "run_folder":
                run_files[key] = str(run_files[key])
        
        all_run_data.append(run_files)
        
    return all_run_data



# Example Usage:
# results = get_paths_to_data("your_config_folder_name")
# for run in results:
#     print(f"Found files for: {run['run_folder']}")
#     print(f" - Metrics: {run['metrics_pkl']}")



