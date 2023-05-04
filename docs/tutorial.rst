Getting Started
================

Let's see how to run the framework. You can try out the framework without having a quantum computer at your disposal by
relying on simulators.

Prerequisites
~~~~~~~~~~~~~

As this framework is implemented in Python ≥ 3.9, you need to install at least Python 3.9 if you don`t have it already installed.
Additionally, we rely on several pip dependencies, which you can install in two ways:

1. Install pip packages manually
2. Use the QUARK installer

To limit the number of packages you need to install, the installer gives you the option to only include a subsection of
QUARK modules.

For this installer to work you need to install the following packages in the first place:
 * inquirer
 * pyyaml

You can select the modules of choice via:

::

   python src/main.py env --install myenv

Of course there is a default option, which will include all available options.

Depending on your installed modules, you will need to install different python packages.
We provide the option to generate a Conda file or a pip requirements file, which you then can install.

You can also install multiple QUARK environments and then switch between them via:

::

   python src/main.py env --activate myenv2

**Note:**  Different modules require different python packages, be sure that you python environment has the necessary packages installed!

To see which environments are installed please use:

::

   python src/main.py env --list

You can also visualize the contents of your QUARK environment:

::


    (quark) %  python src/main.py env --show myenv
    [...]
    Content of the Environment:
    └── TSP
        └── GreedyClassicalTSP
            └── Local


In case you want to use custom modules file (for example to use external modules from other repositories), you can still
use the ``--modules`` option. You can find the documentation in the Dynamic Imports section.


Running a Benchmark
~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   export AWS_PROFILE=quantum_computing
   export HTTP_PROXY=http://username:password@proxy.com:8080
   python src/main.py

``HTTP_PROXY`` is only needed if you are sitting behind proxy.

``AWS_PROFILE`` is only needed if you want to use an aws braket device
(default is quantum_computing). In case no profile is needed in your
case please set ``export AWS_PROFILE=default``.

``AWS_REGION`` is only needed if you need another aws region than
us-east-1. Usually this is specific to the Braket device, so no change
is needed.

Example Run (You need to check at least one option with an ``X`` for the checkbox question):

::

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

Non-Interactive mode
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

Summarizing multiple existing experiments
'''''''''''''''''''''''''''''''''''''''''

You can also summarize multiple existing experiments like this:

::

   python src/main.py --summarize quark/benchmark_runs/2021-09-21-15-03-53 quark/benchmark_runs/2021-09-21-15-23-01

This allows you to generate plots from multiple experiments.


Dynamic Imports
~~~~~~~~~~~~~~~

You can specify the modules that are available in the QUARK framework by specifying a module configuration file with
the option ``-m | --modules``.
This way you can also work with modules that are compatible with QUARK, but are not part of the original QUARK repository.
This also implies that new library dependencies introduced by your modules are needed only if these modules are listed
in the module configuration file.

The module configuration file has to be a json file of the form:
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

The fields ``name`` and ``module`` are mandatory and specify the class name and python module, resp.. ``module``
has to be specified exactly as you would do it within a python import statement. If ``dir`` is specified its
value will be added to the python search path.
In case the class requires some arguments in its constructor they can be defined in the ``args`` dictionary.
In case the class you want use differs from the name you want to show to the user, you can add the name of the class to
the ``class`` argument and leave the user-friendly name in the ``name`` arg.


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

You can save this in a JSON file and then call the framework like:

::

    python src/main.py --modules tsp_example.json

Exploring problem in Jupyter Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also use a jupyter notebook to generate an application instance and create a concrete problem to work on.
Especially while implementing a new mapping or solver, this can be very useful!
