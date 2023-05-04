Developer
==========

The QUARK project welcomes and encourages participation by everyone. This framework is designed to be as modular as possible
to allow developers to easily add new components which are crucial for the continuous development of this framework as
the quantum computing field matures.

Extending the Framework
~~~~~~~~~~~~~~~~~~~~~~~

In this section, we will briefly talk about how one can add a new application, mapping, solver, or device to the
existing ones.

For every component, there is an abstract class that your new class inherits from. These abstract classes provide the required
functions, which are needed for the benchmarking process.

A good starting point is to look at the already existing components.

Adding a new Application
^^^^^^^^^^^^^^^^^^^^^^^^

Alongside adding a new application, you should always add at least one mapping to make the application available for
a solver. Also the new application has to be added to the :code:`app_modules` variable in the :code:`main` function.

Every application has a couple of functions that need to be implemented:
    - :code:`get_solution_quality_unit(self)`: Method to return the unit of the evaluation which is used to make the plots nicer.
      For example for the TSP this could be "Tour Cost".
    - :code:`get_parameter_options(self)`: Method to return the parameters in a dictionary needed to create a concrete problem of an application.
    - :code:`generate_problem(self, config)`: Depending on the application config parameter this method creates a concrete problem and returns it.
    - :code:`validate(self, solution)`: Checks if the solution is a valid solution, needs to return True/False.
    - :code:`evaluate(self, solution)`: Evaluates how good a solution is which allows comparison to other solutions. Should return a float.
    - :code:`save(self, path)`: Method which stores the application instance to a file. Needed for making experiments reproducible.
    - :code:`get_mapping(self, mapping_option)`: Based on :code:`mapping_option` string the function returns an instance of a mapping class.

Optional functions:
    - :code:`process_solution(self, solution)`: Most of the time the solution has to be processed before it can be validated and evaluated.
      This might not be necessary in all cases, so the default is to return the original solution.
    - :code:`regenerate_on_iteration(self, config)`: In case you want to regenerate the application on every iteration this method can be used.

Also, you need to specify the available mapping options :code:`mapping_options` in the constructor of the application class.
With specifying the solvers in :code:`get_parameter_options(self)` and :code:`mapping_options` you decide which mapping is
available for that application.



Example for an Application, which should reside under ``src/applications/myApplication/``:


.. code-block:: python

        import logging
        import os
        from typing import TypedDict
        from time import time

        from applications.Application import *


        class MyApplication(Application):


            def __init__(self):
                super().__init__("MyApplication")
                self.mapping_options = ["Qubo"]

            def get_solution_quality_unit(self) -> str:
                return "my evaluation unit"

            def get_mapping(self, mapping_option):

                if mapping_option == "Qubo":
                    return Qubo()

                else:
                    raise NotImplementedError(f"Mapping Option {mapping_option} not implemented")

            def get_parameter_options(self):

                return {
                    "size": {
                        "values": [3, 4, 6, 8, 10, 14, 16],
                        "description": "What size should your problem be?"
                    }
                }

            class Config(TypedDict):
                size: int


            def generate_problem(self, config: Config, iter_count: int):

                size = config['size']

                self.application = create_problem(size)
                return self.application

            def validate(self, solution):
                start = time() * 1000

                # Check if solution is valid
                if solution is None:
                  logging.error(f"Solution not valid ❌")
                    return False, round(time() * 1000 - start, 3)
                else:
                    logging.info(f"Solution valid ✅ ")
                    return True, round(time() * 1000 - start, 3)

            def evaluate(self, solution):
                start = time() * 1000

                evaluation_metric = calculate_metric(solution)

                return evaluation_metric, round(time() * 1000 - start, 3)

            def save(self, path, iter_count):
                save_your_application(self.application, f"{path}/application.txt")


Adding a new Mapping
^^^^^^^^^^^^^^^^^^^^

As mappings depend highly on the application, you nearly always need to implement a mapping for an application.

Mandatory:
    - :code:`map(self, problem, config)`: Maps the given problem into a specific format a solver can work with. E.g. graph to QUBO.
    - :code:`get_parameter_options(self)`: Method to return the parameters options which can be used to fine tune the mapping.
    - :code:`get_solver(self, solver_option)`: Based on :code:`solver_option` string the function returns an instance of a solver class.

Optional:
    - :code:`reverse_map(self, solution)`: Maps the solution back to the original problem. This might not be necessary in all cases,
      so the default is to return the original solution. This might be needed to convert the solution to a representation needed for validation and evaluation.


Also, you need to specify the available solver options :code:`solver_options` in the constructor of the mapping class.
With specifying the solvers in :code:`get_parameter_options(self)` and :code:`solver_options` you decide which solver is
available for that mapping.



Example for a Mapping, which should reside under ``src/applications/myApplication/mappings``:

.. code-block:: python

        import logging
        from typing import TypedDict

        from applications.Mapping import *
        from solvers.MySolver import MySolver


        class MyMapping(Mapping):

            def __init__(self):
                super().__init__()
                self.solver_options = ["MySolver"]

            def get_parameter_options(self):
                return {
                    "lagrange_factor": {
                        "values": [0.75, 1.0, 1.25],
                        "description": "By which factor would you like to multiply your lagrange?"
                    }
                }

            class Config(TypedDict):
                lagrange_factor: float

            def map(self, graph, config: Config):
                start = time() * 1000
                lagrange = 10
                lagrange_factor = config['lagrange_factor']

                lagrange = lagrange * lagrange_factor

                logging.info(f"Default Lagrange parameter: {lagrange}")

                # Get a QUBO representation of the problem
                q = to_qubo(graph, lagrange)

                return {"Q": q}, round(time() * 1000 - start, 3)

            def get_solver(self, solver_option):

                if solver_option == "MySolver":
                    return MySolver()
                else:
                    raise NotImplementedError(f"Solver Option {solver_option} not implemented")



Adding a new Solver
^^^^^^^^^^^^^^^^^^^^^^^^

Mandatory:
    - :code:`run(self, mapped_problem, device, config, **kwargs)`: Function that solves the mapped problem leveraging the device. Here the actual solving algorithm gets executed.
    - :code:`get_parameter_options(self)`: Method to return the parameters used to fine tune the solver.
    - :code:`get_device(self, device_option)`: Based on :code:`device_option` string the function returns an instance of a device class.


Also, you need to specify the available device options :code:`device_options` in the constructor of the application class.
With specifying the devices in :code:`get_parameter_options(self)` and :code:`device_options` you decide which device is
available for that solver.


Example for a Solver, which should reside under ``src/solvers``:

.. code-block:: python

        from typing import TypedDict

        from devices.MyDevice import MyDevice
        from solvers.Solver import *


        class MySolver(Solver):

            def __init__(self):
                super().__init__()
                self.device_options = ["MyDevice"]

            def get_device(self, device_option):
                if device_option == "MyDevice":
                    return MyDevice()
                else:
                    raise NotImplementedError(f"Device Option {device_option} not implemented")

            def get_parameter_options(self):
                return {
                    "number_of_reads": {
                        "values": [100,250,500,750,1000],
                        "description": "How many reads do you need?"
                    }
                }

            class Config(TypedDict):
                number_of_reads: int

            def run(self, mapped_problem, device_wrapper, config: Config, **kwargs):

                Q = mapped_problem['Q']

                device = device_wrapper.get_device()
                start = time() * 1000
                response = device.solve(Q, num_reads=config['number_of_reads'])
                time_to_solve = round(time() * 1000 - start, 3)

                logging.info(f'MySolver finished in {time_to_solve} ms.')

                return response, time_to_solve



Adding a new Device
^^^^^^^^^^^^^^^^^^^^^^^^

Here you only work with the constructor (device and device_name) to initialize the device.

Example for a Device, which should reside under ``src/devices``:

.. code-block:: python

        from devices.Device import Device

        class MyDevice(Device):

            def __init__(self):
                super().__init__(device_name="MyDevice")
                self.device = MyDevice()

            def get_parameter_options(self):
                return {
                "number_of_cores": {
                    "values": [1,2,3,4],
                    "description": "How many CPU cores do you want to use?"
                    }
                }

            class Config(TypedDict):
                number_of_cores: int

Review Process
~~~~~~~~~~~~~~~

Every Pull Request (PR) is reviewed to help you improve its implementation, documentation, and style. As soon as the PR
is approved by the minimum number of the required reviewer, the PR will be merged to the main branch.
