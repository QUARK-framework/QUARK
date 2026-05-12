import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
from pathlib import Path

# Data visualization: Still does not work properly and needs refinement
def process_quark_data(file_path, title='QUARK Data Visualization', print_summary=True):
    """
    Loads, visualizes, and returns data from .npy or .pkl files.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {os.path.abspath(file_path)}")
        return None

    _, ext = os.path.splitext(file_path)
    
    try:
        # Load logic
        if ext == '.npy':
            data = np.load(file_path, allow_pickle=True)
        elif ext == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            print(f"Unsupported format: {ext}")
            return None
    except Exception as e:
        print(f"Failed to load: {e}")
        return None

    # --- Data Extraction for Listing ---
    # Convert data to a consistent list of (key, value) pairs for the readout
    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, np.ndarray):
        # Handle multidimensional arrays by flattening or taking first row if needed
        flat_data = data.flatten()
        items = list(enumerate(flat_data))
    else:
        items = list(enumerate(data))

    # --- Console Output (The "List" view) ---
    if print_summary:
        print(f"\n{'Index/Key':<20} | {'Value':<10}")
        print("-" * 35)
        for key, val in items:
            # Formatting value to 4 decimal places if it's a float
            val_str = f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val)
            print(f"{str(key):<20} | {val_str:<10}")
        print("-" * 35)

    # --- Visualization ---
    plt.figure(figsize=(10, 5))
    keys = [str(i[0]) for i in items]
    values = [i[1] for i in items]
    
    plt.bar(keys, values, color='skyblue', edgecolor='navy')
    plt.xticks(rotation=45 if len(keys) > 10 else 0)
    plt.xlabel('Index or Key')
    plt.ylabel('Value')
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return data  # Returns the original object for further use


# df = pd.read_pickle(r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\generativemodeling-2026-04-15-09-28-03\benchmark_0\rep_1\data_1.pkl")
# print(df.head())
# print(df.columns)


# best parameters
# process_quark_data(r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\generativemodeling-2026-04-15-09-28-03\benchmark_0\rep_1\best_parameters_1.npy", title='QUARK Results Visualization')

# data
# process_quark_data(r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\generativemodeling-2026-04-15-09-28-03\benchmark_0\rep_1\data_1.pkl", title='QUARK Results Visualization')

# histogram generated
# process_quark_data(r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\generativemodeling-2026-04-15-09-28-03\benchmark_0\rep_1\histogram_generated.npy", title='QUARK Results Visualization')

# histogram train
# process_quark_data(r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\generativemodeling-2026-04-15-09-28-03\benchmark_0\rep_1\histogram_train.npy", title='QUARK Results Visualization')

# record general metrics
# process_quark_data(r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\generativemodeling-2026-04-15-09-28-03\benchmark_0\rep_1\record_gen_metrics_1.pkl", title='QUARK Results Visualization')

# training results -> 
# process_quark_data(r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\generativemodeling-2026-04-15-09-28-03\benchmark_0\rep_1\training_results-1.pkl", title='QUARK Results Visualization')


######

base_path = r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\unsorted\generativemodeling-2026-04-15-09-28-03\benchmark_0\rep_1"

print("QUARK Benchmark Files:")
print("Metrics:", pd.read_pickle(base_path + "\\data_1.pkl").shape)
print("Gen histogram:", np.load(base_path + "\\histogram_generated.npy").shape)
print("Train histogram:", np.load(base_path + "\\histogram_train.npy").shape)
print("Best params:", np.load(base_path + "\\best_parameters_1.npy", allow_pickle=True))

def raw_dump_quark_file(file_path):
    """
    Load and print RAW CONTENTS of QUARK files WITHOUT calculations.
    Just the exact data as stored.
    """
    file_path = Path(file_path)
    print(f"\n{'#'*80}")
    print(f"RAW CONTENTS: {file_path.name}")
    print(f"Path: {file_path.absolute()}")
    print('#'*80)
    
    if not file_path.exists():
        print("FILE NOT FOUND")
        return None
    
    try:
        if file_path.suffix == '.npy':
            data = np.load(file_path, allow_pickle=True)
            print("TYPE:", type(data))
            print("SHAPE:", getattr(data, 'shape', 'N/A'))
            print("DTYPE:", getattr(data, 'dtype', 'N/A'))
            print("RAW DATA:")
            print(repr(data))
            
        elif file_path.suffix == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print("TYPE:", type(data))
            if isinstance(data, pd.DataFrame):
                print("SHAPE:", data.shape)
                print("COLUMNS:", list(data.columns))
                print("\nRAW HEAD (first 10 rows):")
                print(data.head(10).to_dict('records'))
            else:
                print("RAW DATA:")
                print(repr(data))
                
    except Exception as e:
        print("LOAD ERROR:", str(e))
    
    print('#'*80 + '\n')
    return data

# LOAD ALL YOUR FILES - RAW VALUES ONLY
files = [
    r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\unsorted\generativemodeling-2026-04-15-09-28-03\benchmark_0\rep_1\best_parameters_1.npy",
    r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\unsorted\generativemodeling-2026-04-15-09-28-03\benchmark_0\rep_1\data_1.pkl",
    r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\unsorted\generativemodeling-2026-04-15-09-28-03\benchmark_0\rep_1\histogram_generated.npy",
    r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\unsorted\generativemodeling-2026-04-15-09-28-03\benchmark_0\rep_1\histogram_train.npy",
    r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\unsorted\generativemodeling-2026-04-15-09-28-03\benchmark_0\rep_1\record_gen_metrics_1.pkl",
    r"\\wsl.localhost\Ubuntu\home\juana\QUARK-2.1.7_fork\benchmark_runs\unsorted\generativemodeling-2026-04-15-09-28-03\benchmark_0\rep_1\training_results-1.pkl"
]

for f in files:
    raw_dump_quark_file(f)