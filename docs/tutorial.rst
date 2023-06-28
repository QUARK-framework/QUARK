Getting Started
================

Let's see how to run the framework. You can try out the framework without having a quantum computer at your disposal by
relying on simulators.

Prerequisites
~~~~~~~~~~~~~

As this framework is implemented in Python 3.9, you need to install this version of Python if you do not already have it installed. Other versions could cause issues with other dependencies used in the framework. Additionally, we rely on several pip dependencies, which you can install in two ways:

1. Install pip packages manually, or
2. Use the QUARK installer.


For this installer to work, you need to install the following packages in the first place:
* inquirer==3.1.2
* pyyaml==6.0
* seaborn==0.12.2
* networkx==2.8.8
* typing-extensions==4.6.3
* packaging==23.1

To limit the number of packages you need to install, there is an option to only include a subsection of QUARK modules.
You can select the modules of choice via:

::

   python src/main.py env --configure myenv

Of course there is a default option, which will include all available options.

Depending on your configured modules, you will need to install different Python packages. We provide the option to generate a Conda file or a pip requirements file, which you can use to install the required packages.
You can also configure multiple QUARK environments and then switch between them via:

::

   python src/main.py env --activate myenv2

**Note:**  Different modules require different python packages. Be sure that your python environment has the necessary packages installed!

To see which environments are configured, please use

::

   python src/main.py env --list

You can also visualize the contents of your QUARK environment:

::


    (quark) %  python src/main.py env --show myenv
    [...]
    Content of the Environment:
    >-- TSP
        >-- GreedyClassicalTSP
            >-- Local


In case you want to use custom modules files (for example to use external modules from other repositories), you can still
use the ``--modules`` option. You can find the documentation in the Dynamic Imports section.


Running a Benchmark
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   export HTTP_PROXY=http://username:password@proxy.com:8080
   export AWS_PROFILE=quantum_computing
   export AWS_REGION=us-west-1
   python src/main.py

`HTTP_PROXY` is only needed if you have to use a proxy to access AWS.

`AWS_PROFILE` is only needed if you want to use an AWS braket device (default is quantum_computing). In case no profile is needed in your case, please set `export AWS_PROFILE=default`.

`AWS_REGION` is only needed if you need another aws region than us-east-1. Usually this is specific to the Braket device.

Example run (You need to check at least one option with an ``X`` for the checkbox question):

::

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


All used config files, logs and results are stored in a folder in the
``benchmark_runs`` directory.

Non-Interactive Mode
^^^^^^^^^^^^^^^^^^^^

It is also possible to start the script with a config file instead of
using the interactive mode:

::

    python src/main.py --config config.yml

..

   **Note:** This should only be used by experienced users as invalid values will cause the framework to fail!


Example for a config file:

::

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


One handy thing to do is to use the interactive mode once to create a config file.
Then you can change the values of this config file and use it to start the framework.

Summarizing Multiple Existing Experiments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also summarize multiple existing experiments like this:

::

   python src/main.py --summarize quark/benchmark_runs/2021-09-21-15-03-53 quark/benchmark_runs/2021-09-21-15-23-01

This allows you to generate plots from multiple experiments.


Dynamic Imports
~~~~~~~~~~~~~~~

You can specify the modules you want to use in your environment from the list of available modules in the QUARK framework by defining a module configuration file with the option ``-m | --modules``. You can also work with modules that are not part of the original QUARK repository if they are compatible with the rest of the framework. This also implies that new library dependencies introduced by your modules are needed only if these modules are listed in the module configuration file.

The module configuration file has to be a JSON file of the following form:
::

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

The fields ``name`` and ``module`` are mandatory and specify the class name and Python module, respectively. ``module`` has to be equal to the string that would be used as a Python import statement. If ``dir`` is specified, its value will be added to the Python search path. In ``submodules`` you can define a list of subsequent modules that depend on ``module``. In case the class requires some arguments in its constructor, they can be defined in the ``args`` dictionary. In case the name of the class you want to use differs from the name you want to show to users, you can add the name of the class to the ``class`` argument and leave the user-facing name in the ``name`` arg.


An example for this would be:
::

    [
      {
        "name": "TSP",
        "module": "modules.applications.optimization.TSP.TSP",
        "dir": "src",
        "submodules": [
          {
            "name": "GreedyClassicalTSP",
            "module": "modules.solvers.GreedyClassicalTSP",
            "submodules": []
          }
        ]
      }
    ]

You can save this as a JSON file, e.g., tsp_example.json, and then call the framework with the following command:

::

    python src/main.py --modules tsp_example.json

Exploring a Problem in Jupyter Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also use a Jupyter Notebook to generate an application instance and create a concrete problem to work on.
Especially while implementing a new mapping or solver, this can be very useful.
