import unittest
from unittest.mock import patch, MagicMock
import yaml

from modules.Core import Core
from src.ConfigManager import ConfigManager, BenchmarkConfig, ConfigModule


class TestConfigManager(unittest.TestCase):

    def setUp(self):
        self.config_manager = ConfigManager()

    # @patch("src.ConfigManager.inquirer.prompt")
    # @patch("src.ConfigManager.checkbox")
    # @patch("src.ConfigManager._get_instance_with_sub_options")
    # def test_generate_benchmark_configs(self, mock_get_instance, mock_checkbox, mock_prompt):
    #     # Mock application selection
    #     mock_prompt.side_effect = [
    #         {"application": "TestApp"},  # For application selection
    #         {"repetitions": "3"}        # For repetitions
    #     ]
    #     mock_checkbox.return_value = {"submodules": ["TestSubmodule"]}

    #     # Mock application behavior
    #     mock_app_instance = MagicMock()
    #     mock_app_instance.get_parameter_options.return_value = {
    #         "param1": {
    #             "values": [1, 2],
    #             "description": "Select a value for param1",
    #             "exclusive": False,
    #             "custom_input": False,
    #         }
    #     }
    #     mock_app_instance.get_available_submodule_options.return_value = ["TestSubmodule"]
    #     mock_app_instance.get_submodule.return_value = mock_app_instance

    #     # Mock application instance retrieval
    #     mock_get_instance.return_value = mock_app_instance

    #     # Run the function
    #     self.config_manager.generate_benchmark_configs([{"name": "TestApp"}])

    #     # Assertions
    #     self.assertEqual(self.config_manager.config["application"]["name"], "TestApp")
    #     self.assertEqual(self.config_manager.config["repetitions"], 3)
    #     self.assertIn("param1", self.config_manager.config["application"]["config"])
    #     self.assertEqual(self.config_manager.config["application"]["config"]["param1"], [1, 2, 3])

    #     mock_prompt.assert_any_call([unittest.mock.ANY])
    #     mock_checkbox.assert_called_once_with(
    #         key='submodules',
    #         message="What submodule do you want?",
    #         choices=["TestSubmodule"]
    #     )

    # @patch("src.ConfigManager.checkbox")
    # @patch("src.ConfigManager._get_instance_with_sub_options")
    # def test_query_module(self, mock_get_instance, mock_checkbox):
    #     # Mock the checkbox response for "param1" and "submodules"
    #     def checkbox_side_effect(key, message, choices):
    #         if key == "param1":
    #             return {"param1": [1]}
    #         elif key == "submodules":
    #             return {"submodules": ["TestSubmodule"]}
    #         return {}

    #     mock_checkbox.side_effect = checkbox_side_effect

    #     # Mock a module instance
    #     mock_module = MagicMock()
    #     mock_module.get_parameter_options.return_value = {
    #         "param1": {
    #             "values": [1, 2, 3],
    #             "description": "Choose a value for param1",
    #             "exclusive": False,
    #             "custom_input": False
    #         }
    #     }
    #     mock_module.get_available_submodule_options.return_value = ["TestSubmodule"]
    #     mock_module.get_submodule.return_value = mock_module

    #     # Call the method to test
    #     result = self.config_manager.query_module(mock_module, "TestModule")

    #     # Assertions
    #     self.assertEqual(result["name"], "TestModule")
    #     self.assertIn("param1", result["config"])
    #     self.assertEqual(result["config"]["param1"], [1])  # Mocked response
    #     self.assertEqual(len(result["submodules"]), 1)
    #     self.assertEqual(result["submodules"][0]["name"], "TestSubmodule")

    #     # Validate mock calls
    #     mock_checkbox.assert_any_call(
    #         key='param1',
    #         message="(Option for TestModule) Choose a value for param1",
    #         choices=[1, 2, 3]
    #     )
    #     mock_checkbox.assert_any_call(
    #         key='submodules',
    #         message="What submodule do you want?",
    #         choices=["TestSubmodule"]
    #     )
    #     mock_module.get_parameter_options.assert_called_once()
    #     mock_module.get_available_submodule_options.assert_called_once()

    def test_set_config(self):
        config = {
            "application": {
                "name": "TestApp",
                "config": {"param1": 1},
                "submodules": []
            },
            "repetitions": 2
        }
        self.config_manager.set_config(config)

        self.assertEqual(self.config_manager.config["application"]["name"], "TestApp")
        self.assertEqual(self.config_manager.config["repetitions"], 2)

    def test_is_legacy_config(self):
        legacy_config = {"mapping": {"TestMapping": {}}}
        modern_config = {"application": {"name": "TestApp"}, "repetitions": 2}

        self.assertTrue(ConfigManager.is_legacy_config(legacy_config))
        self.assertFalse(ConfigManager.is_legacy_config(modern_config))

    def test_translate_legacy_config_missing_device(self):
        # Mock a legacy config
        config = {
            "application": {"name": "TestApp", "config": {}},
            "repetitions": 1,
            "mapping": {
                "direct": {"solver": [{"name": "DirectSolver", "config": {}, "device": []}]},
                "solver1": {
                    "config": {"param1": "value1"},
                    "solver": [{"name": "Solver1", "config": {"param2": "value2"}, "device": []}],
                },
            },
        }

        # Call the translate_legacy_config method
        translated_config = ConfigManager.translate_legacy_config(config)

        # Assertions to verify the output
        self.assertIn("application", translated_config)
        self.assertIn("repetitions", translated_config)
        self.assertEqual(translated_config["repetitions"], 1)
        self.assertIn("submodules", translated_config["application"])
        self.assertEqual(len(translated_config["application"]["submodules"]), 2)

        # Verify the structure of the submodules
        direct_module = translated_config["application"]["submodules"][0]
        self.assertEqual(len(direct_module["submodules"]), 0)

        solver1_module = translated_config["application"]["submodules"][1]
        self.assertEqual(solver1_module["name"], "solver1")
        self.assertEqual(solver1_module["config"]["param1"], "value1")
        self.assertEqual(len(solver1_module["submodules"]), 1)
        self.assertEqual(solver1_module["submodules"][0]["name"], "Solver1")
        self.assertEqual(solver1_module["submodules"][0]["config"]["param2"], "value2")

    def test_translate_legacy_config_helper(self):
        legacy_solver = {
            "solver": [
                {"name": "SolverA", "config": {}, "device": [{"name": "DeviceA", "config": {}}]}
            ]
        }
        result = ConfigManager.translate_legacy_config_helper(legacy_solver, "solver")

        # Assertions
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "SolverA")
        self.assertEqual(len(result[0]["submodules"]), 1)

    @patch("src.ConfigManager._get_instance_with_sub_options")
    def test_load_config(self, mock_get_instance):
        mock_app_instance = MagicMock()
        mock_get_instance.return_value = mock_app_instance

        config = {
            "application": {
                "name": "TestApp",
                "config": {"param1": 1},
                "submodules": []
            },
            "repetitions": 2
        }
        self.config_manager.set_config(config)

        # Call load_config
        self.config_manager.load_config([{"name": "TestApp"}])

        # Assertions
        self.assertEqual(self.config_manager.application, mock_app_instance)
        mock_get_instance.assert_called_once_with([{"name": "TestApp"}], "TestApp")

    def test_initialize_module_classes(self):
        parent_module = MagicMock(spec=Core)
        parent_module.get_submodule = MagicMock(side_effect=lambda name: MagicMock(name=name))

        config = {
            "name": "TestModule",
            "config": {"param1": "value1"},
            "submodules": [
                {"name": "SubModule1", "config": {}, "submodules": []},
                {"name": "SubModule2", "config": {}, "submodules": []},
            ],
        }

        result = ConfigManager.initialize_module_classes(parent_module, config)
        self.assertEqual(result["name"], "TestModule")
        self.assertEqual(len(result["submodules"]), 2)
        self.assertIn("instance", result)
        parent_module.get_submodule.assert_called_with("TestModule")

    def test_get_config(self):
        mock_config = {"application": {"name": "TestApp"}, "repetitions": 3}
        self.config_manager.config = mock_config
        self.assertEqual(self.config_manager.get_config(), mock_config)

    def test_get_app(self):
        mock_application = MagicMock()
        mock_config = {"application": {"instance": mock_application}}
        self.config_manager.config = mock_config
        self.assertEqual(self.config_manager.get_app(), mock_application)

    def test_get_reps(self):
        mock_config = {"repetitions": 5}
        self.config_manager.config = mock_config
        self.assertEqual(self.config_manager.get_reps(), 5)

    def test_start_create_benchmark_backlog(self):
        mock_application = {
            "name": "TestApp",
            "config": {"param1": [1, 2]},
            "submodules": [
                    {
                        "name": "SubModule1",
                        "config": {"sub_param": [3, 4]},
                        "submodules": [],
                        "instance": MagicMock(),
                    }
            ],
            "instance": MagicMock(),
        }
        self.config_manager.config = {"application": mock_application}
        backlog = self.config_manager.start_create_benchmark_backlog()
        self.assertEqual(len(backlog), 4)
        self.assertEqual(backlog[0]["name"], "TestApp")

    def test_create_benchmark_backlog(self):
        module = {
            "name": "TestModule",
            "config": {"param1": [1, 2]},
            "submodules": [
                    {
                        "name": "SubModule1",
                        "config": {"sub_param": [3, 4]},
                        "submodules": [],
                        "instance": MagicMock(),
                    }
            ],
            "instance": MagicMock(),
        }
        backlog = ConfigManager.create_benchmark_backlog(module)
        self.assertEqual(len(backlog), 4)
        self.assertEqual(backlog[0]["config"]["param1"], 1)
        self.assertEqual(backlog[1]["config"]["param1"], 1)

    def test_save(self):
        mock_config = {"application": {"name": "TestApp"}, "repetitions": 3}
        self.config_manager.config = mock_config
        with patch("builtins.open", unittest.mock.mock_open()) as mocked_open:
            self.config_manager.save("/mock/store/dir")
            mocked_open.assert_called_once_with("/mock/store/dir/config.yml", "w")

    def test_print(self):
        mock_config = {"application": {"name": "TestApp"}, "repetitions": 3}
        self.config_manager.config = mock_config
        with patch("yaml.dump") as mock_yaml_dump:
            self.config_manager.print()
            mock_yaml_dump.assert_called_once_with(mock_config)

    def test_create_tree_figure(self):
        mock_config = {
            "name": "TestApp",
            "submodules": [
                {"name": "SubModule1", "submodules": []},
                {"name": "SubModule2", "submodules": []},
            ],
        }
        self.config_manager.config = {"application": mock_config}
        with patch("matplotlib.pyplot.savefig") as mock_savefig:
            self.config_manager.create_tree_figure("/mock/store/dir")
            mock_savefig.assert_called_once_with("/mock/store/dir/BenchmarkGraph.png", format="PNG")

    # @patch("src.ConfigManager.checkbox")
    # @patch("src.ConfigManager.inquirer.prompt")
    # def test_query_for_config(self, mock_prompt, mock_checkbox):
    #     # Mock user inputs
    #     mock_checkbox.side_effect = [
    #         {"param1": [1, 2, "Custom Input"]},  # Checkbox selection for param1
    #         {"param4": [1, 2, "Custom Range"]}  # Checkbox selection for param4
    #     ]
    #     mock_prompt.side_effect = [
    #         {"custom_input": "custom_value"},  # Custom input for param1
    #         {"start": "0", "stop": "10", "step": "2"},  # Custom range for param4
    #         {"param2": "a"}  # Exclusive selection for param2
    #     ]

    #     param_opts = {
    #         "param1": {
    #             "values": [1, 2, 3],
    #             "description": "Test parameter 1",
    #             "custom_input": True
    #         },
    #         "param2": {
    #             "values": ["a", "b"],
    #             "description": "Test parameter 2",
    #             "exclusive": True
    #         },
    #         "param3": {
    #             "values": [10],
    #             "description": "Single value test"
    #         },
    #         "param4": {
    #             "values": [1, 2, 3],
    #             "description": "Range test",
    #             "allow_ranges": True
    #         }
    #     }

    #     # Call the function
    #     config = ConfigManager._query_for_config(param_opts)

    #     # Assertions for param1
    #     self.assertIn("param1", config)
    #     self.assertEqual(config["param1"], [1, 2, "custom_value"], "Expected combined values for param1.")

    #     # Assertions for param2
    #     self.assertIn("param2", config)
    #     self.assertEqual(config["param2"], ["a"], "Expected exclusive selection for param2.")

    #     # Assertions for param3
    #     self.assertIn("param3", config)
    #     self.assertEqual(config["param3"], [10], "Expected single value to be returned for param3.")

    #     # Assertions for param4
    #     self.assertIn("param4", config)
    #     self.assertEqual(config["param4"], [0.0, 2.0, 4.0, 6.0, 8.0], "Expected generated range for param4.")
