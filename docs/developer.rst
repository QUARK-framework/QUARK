Developer
==========

The QUARK project welcomes and encourages participation by everyone. This framework is designed to be as modular as possible
to allow developers to easily add new components which are crucial for the continuous development of this framework as
the quantum computing field matures.

Extending the Framework
-----------------------

In this section, we will briefly talk about how one can add a new module type to the existing ones.
While with the original release of QUARK one was limited to the modules application, mapping, solver, or device, with
the release of QUARK one can add arbitrary modules.

A good starting point is to look at the already existing components.

Adding a new Module Type
~~~~~~~~~~~~~~~~~~~~~~~~

In the case the already existing module types (see "modules" directory) do not fit your needs, you can create new abstract module definitions, where
you can specify your own requirements.
The only prerequisite is, that the need to fulfill the requirements by the `Core` module, since every module needs to
inherit from this abstract class.

The essential functions of the `Core` module, which can be used by every module/subclass to execute its logic (see figure), are the `preprocess` and the `postprocess` method.
The `preprocess` method executes before the input data is passed on the sub-module, while the `postprocess` method executes before the data is passed onto the parent module.
If no sub-module/parent-module exists, these functions are still executed.

For a concrete application, the `preprocess` step could include generating the problem instance for an optimization problem, while the `postprocess` would include the validation and evaluation of the solution for a given problem

.. image:: benchmark_process.png
  :align: center
  :width: 700
  :alt: Visualization of how the benchmark process is desinged.

You can also implement a new module type that inherits from another abstract module.
So for example we have an `Optimization` class, that specifies the requirements for an optimization application. This
`Optimization` class inherits the requirements from the more general `Application` class.

Every class also needs to implement the `get_requirements` method, where one specifies which imports are used by this class
and which version.
This is needed to provide the user the information which packages need to be installed.

Adding a new Class for an existing Module Type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In this section, we will briefly talk about how one can add a new class to an existing module type.

For every component, there is an abstract class that your new class inherits from.
These abstract classes provide the required functions, which are needed for the benchmarking process.
So for example if you want to implement a new application, you will need to inherit from the abstract `Application` class.


Adding a new Application
^^^^^^^^^^^^^^^^^^^^^^^^

Alongside adding a new application, you should always add at least one submodule to make the application available for
another module (for example a solver). Also the new application has to be added to the :code:`get_default_app_modules` function in the :code:`main` function.

Every application has a couple of functions that need to be implemented:
    - :code:`get_parameter_options(self)`: Method to return the parameters in a dictionary needed to create a concrete problem of an application.
    - :code:`save(self, path)`: Method which stores the application instance to a file. Needed for making experiments reproducible.
    - :code:`get_requirements()`: Which packages are used by this application.
    - :code:`get_default_submodule(self, option)`: Based on :code:`option` string the function returns an instance of a submodule class.
    - :code:`preprocess(self, input_data: any, config: dict, **kwargs)`: Function which is always called and that can be used to pass certain information to the next sub-module.
    - :code:`postprocess(self, input_data: any, config: dict, **kwargs)`:  Function which is always called and that can be used to pass certain information to the next parent-module.


Also, you need to specify the available mapping options :code:`submodule_options` in the constructor of the application class.
With specifying the solvers in :code:`get_default_submodule(self, option)` and :code:`submodule_options` you decide which mapping is
available for that application.
In :code:`get_parameter_options(self)` you can specify which parameters the user can choose from for this module.

There are also some special flags you can set for each parameter:
    - :code:`allow_ranges`: Enabling this feature for your parameter will give the user the option to enter a range for this value. Keep in mind that there is no validation of this user input!
    - :code:`custom_input`: Enabling this feature for your parameter will give the user the option to enter a custom input (text or numbers) for this value. Keep in mind that there is no validation of this user input!
    - :code:`postproc`: Here you can specify a function that should be called and applied to the parameters, should be callable.



Example for an Application, which should reside under ``src/modules/applications/myApplication/``:


.. code-block:: python

        from modules.applications.Application import *


        class MyApplication(Application):


            def __init__(self):
                super().__init__("MyApplication")
                self.submodule_options  = ["submodule1"]

            @staticmethod
            def get_requirements() -> list:
                return [
                    {
                        "name": "networkx",
                        "version": "2.8.8"
                    },
                    {
                        "name": "numpy",
                        "version": "1.24.1"
                    }
                ]

            def get_default_submodule(self, option: str) -> Core:

                if option == "submodule1":
                    return Submodule1()

                else:
                    raise NotImplementedError(f"Submodule Option {option} not implemented")

            def get_parameter_options(self):

                return {
                    "size": {
                        "values": [3, 4, 6, 8, 10, 14, 16],
                        "description": "What size should your problem be?"
                        "allow_ranges: True,
                        "postproc": int
                    },
                    "factor": {
                        "values": [0.75, 1.0, 1.25],
                        "description": "By which factor would you like to multiply your problem?",
                        "custom_input": True,
                        "postproc": float # Since we allow custom input here we need to parse it to float, since input is str
                    }
                }

            class Config(TypedDict):
                size: int
                factor: float

            def preprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):

                # Generate data that gets passed to the next submodule
                start = time() * 1000
                output = self.generate_problem(config)
                return output, round(time() * 1000 - start, 3)

            def postprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):

                # Process data passed to this module from the submodule
                solution_validity, time_to_validation = self.validate(
                    input_data)
                if solution_validity and processed_solution:
                   solution_quality, time_to_evaluation = self.evaluate(processed_solution)

                self.metrics.add_metric_batch({"solution_validity": solution_validity, "solution_quality": solution_quality,
                           "solution": input_data})

                return solution_validity, sum(time_to_validation, time_to_evaluation))



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


Updating the Module Database
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After adding a new module or making a module available for another module you need to update the Module Database stored
under `.settings/module_db.json`. You might also need to update your current QUARK module environment so that your new
modules can be used. You can update this database automatically via `python src/main.py env --createmoduledb`.

**Note:** For `python src/main.py env --createmoduledb` to work you need to have all packages from all modules installed!


Review Process
~~~~~~~~~~~~~~~

Every Pull Request (PR) is reviewed to help you improve its implementation, documentation, and style.
As soon as the PR is approved by the minimum number of the required reviewer, the PR will be merged to the main branch.
