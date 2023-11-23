# QUARK: A Framework for Quantum Computing Application Benchmarking

Quantum Computing Application Benchmark (QUARK) is a framework for orchestrating benchmarks of different industry applications on quantum computers. 
QUARK supports various applications (e.g., TSP, MaxSAT or PVC), different solvers (e.g. Annealing) and different quantum devices (e.g., D-Wave, IonQ), and simulators.
It is designed to be easily extendable in all of its components (Applications, Mappings, Solvers and Devices).

> __Disclaimer:__ This branch contains the original QUARK 1.0 code. So please checkout the "main" branch if you want to use QUARK 2.0.



## Paper
Details about the motivations for the framework can be seen in the accompanying QUARK paper: https://arxiv.org/abs/2202.03028

## Documentation
Documentation with a tutorial and developer guidelines can be found here: https://quark-framework.readthedocs.io

## Prerequisites
As this framework is implemented in Python 3, you need to install Python 3 if you don`t have it already installed. 
Additionally, we rely on several pip dependencies, which you can install in two ways:

1. Install pip packages manually
2. Create new conda env using `environment.yml`:
    ```conda env create -f environment.yml```

> __Note:__ Currently environment.yml is only tested for macOS users!

Some packages such as `pennylane-lightning[gpu]` are not included in this environment file as these work only on specific
hardware! Therefore, these have to be installed manually on demand.

For exporting your environment run:
```
conda env export --no-builds > <environment-name>.yml
```

## Running a Benchmark

```bash 
export AWS_PROFILE=quantum_computing
export HTTP_PROXY=http://username:password@proxy.com:8080 
python src/main.py
```
`HTTP_PROXY` is only needed if you are sitting behind proxy.

`AWS_PROFILE` is only needed if you want to use an aws braket device (default is quantum_computing). 
In case no profile is needed in your case please set `export AWS_PROFILE=default`.

`AWS_REGION` is only needed if you need another aws region than us-east-1.
Usually this is specific to the Braket device, so no change is needed.

Example Run (You need to check at least one option with an ``X`` for the checkbox question):
```
python src/main.py 
[?] What application do you want?: TSP
 > TSP
   PVC
   SAT

[?] (Option for TSP) How many nodes does you graph need?: 
   o 3
   X 4
 > X 5
   o 6

[?] What mapping do you want?: 
   o Ising
 > X Qubo
   o Direct

[?] (Option for Qubo) By which factor would you like to multiply your lagrange?: 
   o 0.75
 > X 1.0
   o 1.25

[?] What Solver do you want for mapping Qubo?: 
 > X Annealer

[?] (Option for Annealer) How many reads do you need?: 
   o 250
 > X 500
   o 750

[?] What Device do you want for solver Annealer?: 
 > X SimulatedAnnealer
   o arn:aws:braket:::device/qpu/d-wave/DW_2000Q_6
   o arn:aws:braket:::device/qpu/d-wave/Advantage_system4

[?] How many repetitions do you want?: 1
2022-02-01 08:25:06,654 [INFO] Created Benchmark run directory /user1/quark/benchmark_runs/tsp-2022-02-01-08-25-06
2022-02-01 08:25:07,133 [INFO] Default Lagrange parameter: 1541836.6666666667
2022-02-01 08:25:07,134 [INFO] Running TSP with config {'nodes': 4} on solver Annealer and device simulatedannealer (Repetition 1/1)
2022-02-01 08:25:07,134 [INFO] Start to measure execution time of <function Annealer.run at 0x1840c7310>
2022-02-01 08:25:07,274 [INFO] Result: {(0, 2): 1, (1, 1): 1, (2, 3): 1, (3, 0): 1}
2022-02-01 08:25:07,274 [INFO] Total execution time of <function Annealer.run at 0x1840c7310>: 139 ms
2022-02-01 08:25:07,275 [INFO] Route found:
 Node 0 →
 Node 2 →
 Node 3 →
 Node 1 🏁
2022-02-01 08:25:07,275 [INFO] All 4 nodes got visited ✅ 
2022-02-01 08:25:07,275 [INFO] Total distance (without return): 807105.0 📏 
2022-02-01 08:25:07,275 [INFO] Total distance (including return): 1516250.0 📏 
2022-02-01 08:25:08,492 [INFO] Saving plot for metric solution_quality
2022-02-01 08:25:08,751 [INFO] Saving plot for metric time_to_solve
2022-02-01 08:25:08,943 [INFO] Saving plot for solver Annealer
```

All used config files, logs and results are stored in a folder in the ```benchmark_runs``` directory.

#### Non-Interactive mode
It is also possible to start the script with a config file instead of using the interactive mode:
```
 python src/main.py --config docs/test_config.yml
```

> __Note:__ This should only be used by experienced users as invalid values will cause the framework to fail!

#### Using your own modules
You can specify the applications, mappers, solvers and devices that the benchmark manager should work with by
specifying a module configuration file with the option '-m | --modules'. This way you can add new modules without
changing the benchmark manager. This also implies that new library dependencies introduced by your modules are
needed only if these modules are listed in the module configuration file.

The module configuration file has to be a json file of the form:
```
[
   {"name":..., "module":..., "dir":..., "mappings":
      [
         {"name":..., "module":..., "dir":..., "solvers":
            [
               {"name":..., "module":..., "dir":..., "devices":
                  [
                     {"name":..., "module":..., "dir":..., "args": {...}, "class": ...},...
                  ]
               },...
            ]
         },...
      ]
   },...
]
```
'name' and 'module' are mandatory and specify the class name and python module, resp.,
'module' has to be specified exactly as you would do it within a python import statement. If 'dir' is specified, its
value will be added to the python search path. 
In case the class requires some arguments in its constructor they can be defined in the 'args' dictionary.
In case the class you want use differs from the name you want to show to the user, you can add the name of the class to
the 'class' argument and leave the user-friendly name in the 'name' arg.

##### Summarizing multiple existing experiments
You can also summarize multiple existing experiments like this:
```
python src/main.py --summarize /Users/user1/quark/benchmark_runs/2021-09-21-15-03-53 /Users/user1/quark/benchmark_runs/2021-09-21-15-23-01
```
This allows you to generate plots from multiple experiments.

### Exploring a problem in Jupyter Notebook
You can also use a jupyter notebook to generate an application instance
and create a concrete problem to work on. Especially while implementing a new solver, this can be very useful!

## License

This project is licensed under [Apache License 2.0](LICENSE).
