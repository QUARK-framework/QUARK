# QUARK: A Framework for Quantum Computing Application Benchmarking

Quantum Computing Application Benchmark (QUARK) is a framework for orchestrating benchmarks of different industry applications on quantum computers. 
QUARK supports various applications such as the traveling salesperson problem (TSP), the maximum satisfiability (MaxSAT) problem, robot path optimization in the PVC sealing use case 
as well as new additions like the Maximum Independent Set (MIS), Set Cover Problem (SCP) and Auto Carrier Loading (ACL).
It also features different solvers (e.g., simulated /quantum annealing and the quantum approximate optimization algorithm (QAOA)), quantum devices (e.g., IonQ and Rigetti), and simulators.
It is designed to be easily extendable in all of its components: applications, mappings, solvers, devices, and any other custom modules.

## Publications
Details about the motivations for the original framework can be found in the [accompanying QUARK paper from Finžgar et al](https://arxiv.org/abs/2202.03028).
Even though the architecture changes significantly from QUARK 1.0 to the current version, the guiding principles still remain. The most recent publication from [Kiwit et al.](https://arxiv.org/abs/2308.04082) provides an updated overview of the functionalities and quantum machine learning features of QUARK.

## Documentation
Documentation with a tutorial and developer guidelines can be found here: https://quark-framework.readthedocs.io/en/dev/.

## Prerequisites

Using all of QUARK's features requires a lot of dependencies and increases QUARK's required starting time. To remedy that, QUARK manages its own module system alongside Python environments and keeps both in sync. 
To use QUARK correctly and conveniently, we recommend using the uv package manager as described below. Of course, you can still set up QUARK manually. We also provide a description for that option.

### With uv
#### Step 1: Install uv
https://docs.astral.sh/uv/getting-started/installation/

#### Step 2: Run the setup script and choose which features to use
```bash
bash setup.sh
```

#### Step 3: Run QUARK
```bash
uv run python src/main.py
```
> __Note:__ This step might take a little longer when run the first time after a setup.

#### Step 4 (Optional): Run the setup script again to switch dependencies
```bash
bash setup.sh
```

### Without uv (all modules enabled)
#### Step 1: Install all dependencies
Make sure that your python version is 3.12, then install all dependencies
```bash
pip install -r .settings/requirements_full.txt
```

#### Step 2: Run QUARK
```bash
python src/main.py
```

<br>

> __Note:__ It is still possible to only use a subset of QUARK's modules without also using uv. The process is explained in the [documentation](https://quark-framework.readthedocs.io/en/dev/tutorial.html).

## Git Large File Storage (LFS)
QUARK stores data and config files using **Git LFS**. If you are contributing to this project or cloning this repository, ensure that you have Git LFS installed and configured to manage large files effectively.

### Installing Git LFS
Install Git LFS by following the instructions on [Git LFS](https://git-lfs.com/):
  - On Linux/macOS
    ```bash
    git lfs install
    ```
  - On Windows. Download and install Git LFS from the [official page](https://git-lfs.com/)

## Running a Benchmark

#### Setting Global Variables
```bash
export HTTP_PROXY=http://username:password@proxy.com:8080 
export AWS_PROFILE=quantum_computing
export AWS_REGION=us-west-1
```
`HTTP_PROXY` is only needed if you have to use a proxy to access AWS.

`AWS_PROFILE` is only needed if you want to use an AWS braket device (default is quantum_computing). In case no profile is needed in your case, please set `export AWS_PROFILE=default`.

`AWS_REGION` is only needed if you need another aws region than us-east-1. Usually this is specific to the Braket device.

#### Interactive Mode
You can run QUARK using
```
python src/main.py
```
This will initiate an interactive configuration mode to describe what you want to benchmark. After finishing the configuration, the benchmark run begins automatically. The results and the configuration ```config.yml``` file are saved with a timestamp in ```benchmark_runs```.

#### Non-Interactive Mode
It is also possible to start the script with a config file instead of using the interactive mode:
```
 python src/main.py --config config.yml
```

> __Note:__ This should only be used by experienced users as invalid values will cause the framework to fail!

Example for a config file:

```
application:
  config:
    nodes:
    - 3
  name: TSP
  submodules:
  - config: {}
    name: GreedyClassicalTSP
    submodules:
    - config: {}
      name: Local
      submodules: []
repetitions: 1
```

### Running Specific Modules

If you want to run specific modules, use the preconfigured YAML files under ```tests/config/valid/```. 

For example:

```python src/main.py -c tests/config/valid/TSP.yml```

Replace ```TSP.yml``` with the desired module configuration (e.g., ```MIS.yml```, ```generativemodeling.yml```, etc.)
> __Note:__ This should only be used by experienced users as invalid values will cause the framework to fail!


Example run (you need to check at least one option with an ``X`` for the checkbox question):
```
(quark) % python src/main.py 
[?] What application do you want?: TSP
   PVC
   SAT
 > TSP
   ACL
   MIS
   SCP
   GenerativeModeling

2024-10-09 15:05:52,610 [INFO] Import module modules.applications.optimization.tsp.tsp
[?] (Option for TSP) How many nodes does you graph need?:
 > [X] 3
   [ ] 4
   [ ] 6
   [ ] 8
   [ ] 10
   [ ] 14
   [ ] 16
   [ ] Custom Range

[?] What submodule do you want?:
   [ ] Ising
   [ ] Qubo
 > [X] GreedyClassicalTSP
   [ ] ReverseGreedyClassicalTSP
   [ ] RandomTSP

2024-10-09 15:06:20,897 [INFO] Import module modules.solvers.greedy_classical_tsp
2024-10-09 15:06:20,933 [INFO] Skipping asking for submodule, since only 1 option (Local) is available.
2024-10-09 15:06:20,933 [INFO] Import module modules.devices.Local
2024-10-09 15:06:20,946 [INFO] Submodule configuration finished
[?] How many repetitions do you want?: 1P
2024-10-09 15:07:11,573 [INFO] Import module modules.applications.optimization.tsp.tsp
2024-10-09 15:07:11,573 [INFO] Import module modules.solvers.greedy_classical_tsp
2024-10-09 15:07:11,574 [INFO] Import module modules.devices.Local
2024-10-09 15:07:12,194 [INFO] [INFO] Created Benchmark run directory /Users/user1/quark/benchmark_runs/tsp-2024-10-09-15-07-11
2024-10-09 15:07:12,194 [INFO] Codebase is based on revision 1d9d17aad7ddff623ff51f62ca3ec2756621c345 and has no uncommitted changes
2024-10-09 15:07:12,195 [INFO] Running backlog item 1/1, Iteration 1/1:
2024-10-09 15:07:12,386 [INFO] Route found:
 Node 0 ->
 Node 2 ->
 Node 1
2024-10-09 15:07:12,386 [INFO] All 3 nodes got visited
2024-10-09 15:07:12,386 [INFO] Total distance (without return): 727223.0
2024-10-09 15:07:12,386 [INFO] Total distance (including return): 1436368.0
2024-10-09 15:07:12,386 [INFO]
2024-10-09 15:07:12,386 [INFO] ==== Run backlog item 1/1 with 1 iterations - FINISHED:1 ====
2024-10-09 15:07:12,387 [INFO]
2024-10-09 15:07:12,387 [INFO] =============== Run finished ===============
2024-10-09 15:07:12,387 [INFO]
2024-10-09 15:07:12,387 [INFO] ================================================================================
2024-10-09 15:07:12,387 [INFO] ====== Run 1 backlog items with 1 iterations - FINISHED:1
2024-10-09 15:07:12,387 [INFO] ================================================================================
2024-10-09 15:07:12,395 [INFO]
2024-10-09 15:07:12,400 [INFO] Saving 1 benchmark records to /Users/user1/QUARK/benchmark_runs/tsp-2024-10-09-15-07-11/results.json
2024-10-09 15:07:12,942 [INFO] Finished creating plots.
2024-10-09 15:07:12,943 [INFO] ============================================================ 
2024-10-09 15:07:12,944 [INFO] ====================  QUARK finished!   ====================
2024-10-09 15:07:12,944 [INFO] ============================================================

```

All used config files, logs and results are stored in a folder in the ```benchmark_runs``` directory.

### Interrupt/resume
The processing of backlog items may get interrupted in which case you will see something like
```
2024-03-13 10:25:20,201 [INFO] ================================================================================
2024-03-13 10:25:20,201 [INFO] ====== Run 3 backlog items with 10 iterations - FINISHED:15 INTERRUPTED:15
2024-03-13 10:25:20,201 [INFO] ====== There are interrupted jobs. You may resume them by running QUARK with
2024-03-13 10:25:20,201 [INFO] ====== --resume-dir=benchmark_runs\tsp-2024-03-13-10-25-19
2024-03-13 10:25:20,201 [INFO] ================================================================================
```
This happens if you press CTRL-C or if some QUARK module does its work asynchronously, e.g. by submitting its job to some 
batch system. Learn more about how to write asynchronous modules in the [developer guide](https://quark-framework.readthedocs.io/en/dev/developer.html).
You can resume an interrupted QUARK run by calling: 
```
python src/main.py --resume-dir=<result-dir>
```
Note that you can copy/paste the ```--resume-dir``` option from the QUARK output as shown in the above example.

### Run as Container
We also support the option to run the framework as a container.
After making sure your docker daemon is running, you can run the container:
```
docker run -it --rm ghcr.io/quark-framework/quark
```

> __Note__: ARM builds are (temporarily) removed in release 2.1.3 because pyqubo 1.5.0 is unavailable for this platform
> at the moment. This means if you want to run QUARK as a container on a machine with a chip from this 
> [list](https://en.wikipedia.org/wiki/List_of_ARM_processors) you might face problems. Please feel free to 
> [open an issue](https://github.com/QUARK-framework/QUARK/issues/new), so we can work on a tailored workaround until 
> the latest version of pyqubo is available on ARM platforms.

You can also build the docker image locally like:
``` 
docker build -t ghcr.io/quark-framework/quark .
```

In case you want to use a config file you have to add it to the docker run command: 
```
-v /Users/alice/desktop/my_config.yml:/my_config.yml
```
`/Users/alice/desktop/my_config.yml` specifies the QUARK config file on your local machine.
Then you can run the docker container with the config:
```
docker run -it --rm  -v /Users/alice/desktop/my_config.yml:/my_config.yml ghcr.io/quark-framework/quark --config my_config.yml
```

In case you want to access the benchmark run folder afterwards, you can attach a volume to the run command:
```
-v /Users/alice/desktop/benchmark_runs:/benchmark_runs/
```
The results of the benchmark run are then stored to a new directory in `/Users/alice/desktop/benchmark_runs`.

In case you have local proxy settings you can add the following flags to the run command:

```
-e http_proxy=$http_proxy -e https_proxy=$https_proxy -e HTTP_PROXY=$HTTP_PROXY -e HTTPS_PROXY=$HTTPS_PROXY
```

AWS credentials can be mounted to the run command like:
```
-v $HOME/.aws/:/root/.aws:ro
```



#### Summarizing Multiple Existing Experiments
You can also summarize multiple existing experiments like this:
```
python src/main.py --summarize quark/benchmark_runs/2021-09-21-15-03-53 quark/benchmark_runs/2021-09-21-15-23-01
```
This allows you to generate plots from multiple experiments.


## License

This project is licensed under [Apache License 2.0](LICENSE).
