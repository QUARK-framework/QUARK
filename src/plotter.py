#  Copyright 2021 The QUARK Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib
from matplotlib import pyplot as plt
from collections import defaultdict
import logging

matplotlib.rcParams['savefig.dpi'] = 300
sns.set(style="darkgrid")


class Plotter:
    """
    Plotter class which generates some general plots.
    """

    @staticmethod
    def visualize_results(results: list[dict], store_dir: str) -> None:
        """
        Function to plot the execution times of the benchmark.

        :param results: Dict containing the results
        :param store_dir: Directory where the plots are stored
        """
        if results is None or len(results) == 0:
            logging.info("Nothing to plot since results are empty.")
            return

        processed_results_with_application_score = []
        processed_results_rest = []
        required_application_score_keys = [
            "application_score_value", "application_score_unit", "application_score_type"
        ]
        application_name = None
        application_axis = None
        static_keys, changing_keys = Plotter._get_config_keys(results)
        for result in results:
            application_name = result["module"]["module_name"]
            if len(changing_keys) == 1:
                # If only 1 config item changes, we use its value for application_config
                application_axis = changing_keys[0]
                application_config = result['module']['module_config'][changing_keys[0]]
            else:
                # If multiple config items change, we stringify them
                application_axis = f"{application_name} Config"
                application_config = ', '.join(
                    [f"{key}: {value}" for (key, value) in sorted(result["module"]["module_config"].items(),
                                                                  key=lambda key_value_pair:
                                                                  key_value_pair[0]) if key not in static_keys]
                )
            if len(static_keys) > 0:
                # Include the static items in the axis name
                application_axis += "(" + ', '.join(
                    [f"{key}: {result['module']['module_config'][key]}" for key in static_keys]
                ) + ")"

            processed_item = Plotter._extract_columns({
                "benchmark_backlog_item_number": result["benchmark_backlog_item_number"],
                "total_time": result["total_time"],
                "application_config": application_config
            }, result["module"])

            if all(k in result["module"] for k in required_application_score_keys):
                # Check if all required keys are present to create application score plots
                for k in required_application_score_keys:
                    processed_item[k] = result["module"][k]
                processed_results_with_application_score.append(processed_item)
            else:
                processed_results_rest.append(processed_item)

        if len(processed_results_with_application_score) > 0:
            logging.info("Found results with an application score, generating according plots.")
            Plotter.plot_application_score(
                application_name, application_axis, processed_results_with_application_score, store_dir
            )

        Plotter.plot_times(
            application_name, application_axis,
            [*processed_results_with_application_score, *processed_results_rest],
            store_dir, required_application_score_keys
        )

        # Plotter.plot_all_metrics(results, store_dir)  # TODO

        logging.info("Finished creating plots.")

    @staticmethod
    def plot_times(application_name: str, application_axis: str, results: list[dict], store_dir: str,
                   required_application_score_keys: list) -> None:
        """
        Function to plot execution times of the different modules in a benchmark.

        :param application_name: Name of the application
        :param application_axis: Name of the application axis
        :param results: Dict containing the results
        :param store_dir: Directory where the plots are stored
        :param required_application_score_keys: List of keys which have to be present to calculate an application score
        """

        df = pd.DataFrame.from_dict(results)
        df = df.fillna(0.0).infer_objects(copy=False)
        df_melt = df.drop(df.filter(["application_config", "config_combo", "total_time",
                                     *required_application_score_keys]), axis=1)
        df_melt = pd.melt(frame=df_melt, id_vars='benchmark_backlog_item_number', var_name='module_config',
                          value_name='time')

        # This plot shows the execution time of each module
        ax = sns.barplot(x="benchmark_backlog_item_number", y="time", data=df_melt, hue="module_config")
        plt.title(application_name)
        # Put the legend out of the figure
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Modules used")
        ax.set(xlabel="benchmark run ID", ylabel='execution time of module (ms)')
        plt.savefig(f"{store_dir}/time_by_module.pdf", dpi=300, bbox_inches='tight')
        logging.info(f"Saved {f'{store_dir}/time_by_module.pdf'}.")
        plt.clf()

        # This plot shows the total time of a benchmark run
        ax = sns.barplot(x="application_config", y="total_time", data=df, hue="config_combo")
        ax.set(xlabel=application_axis, ylabel="total execution time (ms)")
        # Put the legend out of the figure
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Modules used")
        plt.title(application_name)
        plt.sca(ax)
        # If column values are very long and of type string rotate the ticks
        if (pd.api.types.is_string_dtype(df.application_config.dtype) or pd.api.types.is_object_dtype(
                df.application_config.dtype)) and df.application_config.str.len().max() > 10:
            plt.xticks(rotation=90)
        plt.savefig(f"{store_dir}/total_time.pdf", dpi=300, bbox_inches='tight')
        logging.info(f"Saved {f'{store_dir}/total_time.pdf'}.")
        plt.clf()

    @staticmethod
    def plot_application_score(application_name: str, application_axis: str, results: list[dict],
                               store_dir: str) -> None:
        """
        Function to create plots showing the application score.

        :param application_name: Name of the application
        :param application_axis: Name of the application axis
        :param results: Dict containing the results
        :param store_dir: Directory where the plots are stored
        """
        df = pd.DataFrame.from_dict(results)
        application_score_units = df["application_score_unit"].unique()
        count_invalid_rows = pd.isna(df['application_score_value']).sum()

        if count_invalid_rows == len(df):
            logging.info("All results have an invalid application score, skipping plotting.")
            return
        else:
            logging.info(f"{count_invalid_rows} out of {len(df)} benchmark runs have an invalid application score.")

        if len(application_score_units) != 1:
            logging.warning(
                f"Found more or less than exactly 1 application_score_unit in {application_score_units}."
                f" This might lead to incorrect plots!"
            )

        ax = sns.barplot(x="application_config", y="application_score_value", data=df, hue="config_combo")
        ax.set(xlabel=application_axis, ylabel=application_score_units[0])
        # Put the legend out of the figure
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Modules used")
        ax.text(
            1.03, 0.5,
            f"{len(df) - count_invalid_rows}/{len(df)} runs have a valid \napplication score",
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox={"boxstyle": "round", "alpha": 0.15}
        )
        plt.title(application_name)

        plt.sca(ax)
        # If column values are very long and of type string, rotate the ticks
        if (pd.api.types.is_string_dtype(df.application_config.dtype) or pd.api.types.is_object_dtype(
                df.application_config.dtype)) and df.application_config.str.len().max() > 10:
            plt.xticks(rotation=90)

        plt.savefig(f"{store_dir}/application_score.pdf", dpi=300, bbox_inches='tight')
        logging.info(f"Saved {f'{store_dir}/application_score.pdf'}.")
        plt.clf()

    @staticmethod
    def _get_config_keys(results: list[dict]) -> tuple[list, list]:
        """
        Function that extracts config keys.

        :param results: Results of a benchmark run
        :return: Tuple with list of static keys and list of changing keys
        """
        static_keys = []
        changing_keys = []
        helper_dict = defaultdict(list)
        # Try to find out which key in the dict change
        for result in results:  # you can list as many input dicts as you want here
            d = result["module"]["module_config"]
            for key, value in d.items():
                helper_dict[key].append(value)
                helper_dict[key] = list(set(helper_dict[key]))

        for key, value in helper_dict.items():
            if len(value) == 1:
                static_keys.append(key)
            else:
                changing_keys.append(key)

        return static_keys, changing_keys

    @staticmethod
    def _extract_columns(config: dict, rest_result: dict) -> dict:
        """
        Function to extract and summarize certain data fields like the time spent in every module
        from the nested module chain.

        :param config: Dictionary containing multiple data fields like the config of a module
        :param rest_result: Rest of the module chain
        :return: Extracted data
        """
        if rest_result:
            module_name = rest_result["module_name"]
            for key, value in sorted(rest_result["module_config"].items(),
                                     key=lambda key_value_pair: key_value_pair[0]):
                module_name += f", {key}: {value}"

            config_combo = config.pop("config_combo") + "\n" + module_name if "config_combo" in config else ""
            return Plotter._extract_columns(
                {
                    **config,
                    "config_combo": config_combo,
                    module_name: rest_result["total_time"]
                    if module_name not in config else config[module_name] + rest_result["total_time"]
                },
                rest_result["submodule"]
            )

        return config

    @staticmethod
    def make_radar_chart(name, subname, store_dir, stats, attribute_labels):
        """
        name: Plot title,
        subname: Plot subtitle (relevant experiment info)
        store_dir: folder where to store plot,
        stats: metric values,
        attribute_labels: metric names
        """

        assert len(stats) == len(attribute_labels), "labels and values to plot do not have the same length!"
        attribute_labels.append('')

        markers = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # can be any length
        labels = np.array(attribute_labels)

        angles = np.linspace(0, 2 * np.pi, len(labels) - 1, endpoint=False)
        stats = np.concatenate((stats, [stats[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        ax.plot(angles, stats, 'o-', linewidth=2)
        ax.fill(angles, stats, alpha=0.25)
        ax.grid(True)
        # ax.grid(c="black")
        ax.spines['polar'].set_visible(False)
        ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels[:-1])
        ax.set_ylim([0, 1])
        ax.plot(np.linspace(0, 2 * np.pi, 500), np.ones(500), color="k", linewidth=1)

        plt.yticks(markers)
        plt.title(subname)
        plt.suptitle(name, y=1.02)

        fig.savefig(f"{store_dir}/metrics_collection_test.pdf", dpi=300,
                    bbox_inches='tight')  # dpi=300, bbox_inches='tight'
        fig.savefig(f"{store_dir}/metrics_collection_test.png", dpi=300, bbox_inches='tight')
        return

    @staticmethod
    def plot_all_metrics(results: list[dict], store_dir: str) -> None:

        # Total time
        overall_time, overall_time_unit = results[0]["total_time"], results[0]["total_time_unit"]
        m_name = results[0]["module"]["module_name"]

        # Use-case specific results
        if m_name == "GenerativeModeling":
            dataset_name = results[0]["module"]["submodule"]["module_name"]
            kl_best = results[0]["module"]["submodule"]["KL_best"]
            gen_metrics = results[0]["module"]["submodule"]["generalization_metrics"]  # TODO
            precision = gen_metrics["precision"]
            fidelity = gen_metrics["fidelity"]

            quantum_module = results[0]["module"]["submodule"]["submodule"]["submodule"]["submodule"]
            if "population_size" in quantum_module["module_config"]:
                pop_size = quantum_module["module_config"]["population_size"]
            else:
                pop_size = None
            max_evaluations = quantum_module["module_config"]["max_evaluations"]
            quantum_time, quantum_time_unit = quantum_module["total_time"], quantum_module["total_time_unit"]
            if overall_time_unit == quantum_time_unit:
                time_ratio = float(quantum_time) / (float(overall_time))
            else:
                print('Hybrid module time and overall time have different time units, please check.')
                time_ratio = 0.
            ent = quantum_module["meyer-wallach"]
            expr = quantum_module["expressibility_jsd"]

            metrics_vector = [time_ratio, precision, fidelity, expr, ent]
            metrics_names = ['Time ratio', 'Precision', 'Fidelity', 'Expressibility', 'Entanglement']
            plt_title = "GenerativeModeling"
            info_str = f"Data: {dataset_name}, Population size: {pop_size}, Max. Evaluations: {max_evaluations}"

        elif m_name == "Classification":
            # Results
            quantum_module = results[0]["module"]["submodule"]["submodule"]
            quantum_time, quantum_time_unit = quantum_module["total_time"], quantum_module["total_time_unit"]
            assert quantum_module["module_name"] == "Hybrid", f"Module name is not hybrid but {
                quantum_module['module_name']}. Are you sure this is correct?"
            if overall_time_unit == quantum_time_unit:
                time_ratio = float(quantum_time) / (float(overall_time))
            else:
                print('Hybrid module time and overall time have different time units, please check.')
                time_ratio = 0.

            # Experiment info
            n_epochs = quantum_module["module_config"]["n_epochs"]
            setup_info = results[0]["module"]["submodule"]["module_config"]
            n_classes = setup_info["n_classes"]
            dataset = setup_info["data_set"]

            metrics_vector = [
                time_ratio,
                quantum_module["train_accuracy"],
                quantum_module["val_accuracy"],
                quantum_module["expressibility_jsd"],
                quantum_module["meyer-wallach"]]
            metrics_names = ['Time ratio', 'Acc_train', 'Acc_test', 'Expressibility', 'Entanglement']
            plt_title = f"QNN: {dataset} (Cls={n_classes})"
            info_str = f"Epochs: {n_epochs}, Noise: {
                setup_info['noise_sigma']}, Images: {
                setup_info['n_images_per_class']}"

        else:
            print(f"{m_name} is not implemented for plotting, no radar plot generated.")
            return

        # Make metrics plot
        Plotter.make_radar_chart(plt_title, info_str, store_dir, metrics_vector, metrics_names)

        logging.info(f"Saved {f'{store_dir}/metrics_collection_test.pdf'}.")
