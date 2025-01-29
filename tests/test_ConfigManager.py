import unittest
from unittest.mock import patch, MagicMock

from quark.modules.Core import Core
from quark.ConfigManager import ConfigManager


class TestConfigManager(unittest.TestCase):

    def setUp(self):
        self.config_manager = ConfigManager()

    @patch("src.ConfigManager.inquirer.prompt")
    @patch("src.ConfigManager.checkbox")
    def test_query_module(self, mock_checkbox, mock_prompt):
        # Mock responses for checkbox and prompt
        mock_checkbox.return_value = {"param1": [1, 2]}  # Simulates a user selecting 1 and 2
        mock_prompt.return_value = {"param2": "a"}  # Simulates a user selecting 'a' for param2

        param_opts = {
            "param1": {"values": [1, 2, 3], "description": "Test parameter 1"},
            "param2": {"values": ["a", "b"], "description": "Test parameter 2", "exclusive": True},
        }

        config = ConfigManager._query_for_config(param_opts)

        # Assert the results
        self.assertEqual(config["param1"], [1, 2])

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

    @patch("src.ConfigManager.inquirer.prompt")
    @patch("src.ConfigManager.checkbox")
    def test_query_for_config(self, mock_checkbox, mock_prompt):
        mock_checkbox.return_value = {"param1": [1, 2, "Custom Input"]}
        mock_prompt.side_effect = [{"custom_input": "custom_value"}]
        param_opts = {
            "param1": {"values": [1, 2, 3], "description": "Test parameter 1", "custom_input": True}
        }
        config = ConfigManager._query_for_config(param_opts)
        self.assertIn("param1", config)
        self.assertEqual(config["param1"], [1, 2, "custom_value"])
