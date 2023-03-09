Benchmark Data Analysis
========================

Usually, the automatically generated plots are not enough for detailed analysis of experiments.
Therefore, we advise you to use the tool of your choice for working with the ``result.json`` generated during an experiment
run to analyze your data more thoroughly.


Following is a simple example of how to load the json file into a pandas dataframe which can then be used for further analysis.

Python Example
~~~~~~~~~~~~~~

.. code-block:: python

        import json

        # Let`s read in the results
        filename = "benchmark_runs/tsp-2023-03-13-15-31-17/results.json"
        with open(filename) as f:

            results = json.load(f)