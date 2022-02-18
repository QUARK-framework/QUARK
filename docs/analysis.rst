Benchmark Data Analysis
========================

Usually, the automatically generated plots are not enough for detailed analysis of experiments.
Therefore, we advise you to use the tool of your choice for working with the ``result.csv`` generated during an experiment
run to analyze your data more thoroughly.


Following is a simple example of how to load the CSV file into a pandas dataframe which can then be used for further analysis.

Python Example
~~~~~~~~~~~~~~

.. code-block:: python

        import pandas as pd
        import json

        # Let`s read in the results
        filename = "benchmark_runs/tsp-2022-02-01/results.csv"
        df = pd.read_csv(filename, index_col=0, encoding="utf-8")

        # We need to load the columns correctly since they contain json
        df['application_config'] = df.apply(lambda row: json.loads(row["application_config"]), axis=1)
        df['solver_config'] = df.apply(lambda row: json.loads(row["solver_config"]), axis=1)
        df['mapping_config'] = df.apply(lambda row: json.loads(row["mapping_config"]), axis=1)

        # Now we can do plots or other analysis using that df