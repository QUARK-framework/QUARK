Benchmark Data Analysis
========================

If the plots generated in BenchmarkManager.vizualize_results are not sufficient, a more thorough analysis of the benchmark experiment should be run with a tool of your choice using the generated results in the results.json file. This file is stored in a dedicated location in the benchmark_runs directory. The name of the location starts with the name of the application evaluated in the benchmark and ends with the timestamp of the experiment. Below is a simple example of how to load the JSON file into a pandas DataFrame, which can be used for further analysis.

Python Example
~~~~~~~~~~~~~~

.. code-block:: python

        import json

        # Load the results
        filename = "benchmark_runs/tsp-2023-03-13-15-31-17/results.json"
        with open(filename) as f:

            results = json.load(f)
