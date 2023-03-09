# QUARK: A Framework for Quantum Computing Application Benchmarking

Quantum Computing Application Benchmark (QUARK) is a framework for orchestrating benchmarks of different industry applications on quantum computers. 
QUARK supports various applications (e.g., TSP, MaxSAT or PVC), different solvers (e.g. Annealing) and different quantum devices (e.g., IonQ), and simulators.
It is designed to be easily extendable in all of its components (Applications, Mappings, Solvers, Devices or other custom module definitions).

## Paper
Details about the motivations for the framework can be seen in the accompanying QUARK paper: https://arxiv.org/abs/2202.03028. 
Even though the architecture changed quite significantly with the 2.0 release of QUARK, the guiding principles still remain.

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
(quark) % python src/main.py 
[?] What application do you want?: TSP
   PVC
   SAT
 > TSP

2023-03-21 09:18:36,440 [INFO] Import module modules.applications.optimization.TSP.TSP
[?] (Option for TSP) How many nodes does you graph need?:
 > [X] 3
   [ ] 4
   [ ] 6
   [ ] 8
   [ ] 10
   [ ] 14
   [ ] 16

[?] What submodule do you want?:
   [ ] Ising
   [ ] Qubo
 > [X] GreedyClassicalTSP
   [ ] ReverseGreedyClassicalTSP
   [ ] RandomTSP

2023-03-21 09:18:49,563 [INFO] Skipping asking for submodule, since only 1 option (Local) is available.
2023-03-21 09:18:49,566 [INFO] Submodule configuration finished
[?] How many repetitions do you want?: 1
2023-03-21 09:18:50,577 [INFO] Import module modules.applications.optimization.TSP.TSP
2023-03-21 09:18:50,948 [INFO] Created Benchmark run directory /Users/user1/QUARK/benchmark_runs/tsp-2023-03-21-09-18-50
2023-03-21 09:18:51,025 [INFO] Codebase is based on revision 075201825fa71c24b5567e1290966081be7dbdc0 and has some uncommitted changes
2023-03-21 09:18:51,026 [INFO] Running backlog item 1/1, Iteration 1/1:
2023-03-21 09:18:51,388 [INFO] Route found:
 Node 0 ->
 Node 2 ->
 Node 1
2023-03-21 09:18:51,388 [INFO] All 3 nodes got visited
2023-03-21 09:18:51,388 [INFO] Total distance (without return): 727223.0
2023-03-21 09:18:51,388 [INFO] Total distance (including return): 1436368.0
2023-03-21 09:18:51,389 [INFO]
2023-03-21 09:18:51,389 [INFO]  ============================================================
2023-03-21 09:18:51,389 [INFO]
2023-03-21 09:18:51,389 [INFO] Saving 1 benchmark records to /Users/user1/QUARK/benchmark_runs/tsp-2023-03-21-09-18-50/results.json
2023-03-21 09:18:51,746 [INFO] Finished creating plots.
2023-03-21 09:18:51,746 [INFO]  ============================================================
2023-03-21 09:18:51,746 [INFO]  ====================  QUARK finished!   ====================
2023-03-21 09:18:51,746 [INFO]  ============================================================

```

All used config files, logs and results are stored in a folder in the ```benchmark_runs``` directory.

#### Non-Interactive mode
It is also possible to start the script with a config file instead of using the interactive mode:
```
 python src/main.py --config test_config.yml
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
  {"name":..., "module":..., "dir":..., "submodules":
     [
        {"name":..., "module":..., "dir":..., "submodules":
           [
              {"name":..., "module":..., "dir":..., "args": {...}, "class": ..., submodules":
                 []
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
