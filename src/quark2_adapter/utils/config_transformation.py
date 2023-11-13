""" Module to transform on the fly the QUARK 1.0 
app-modules as well as the runtime config.yml to match
the schema of QUARK 2.0 after migration
"""
#  Copyright 2023 science + computing ag
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#

from copy import deepcopy
import logging


def recursive_app_modules_convert(obj) -> None:
    """replace some keywords inside of e.g. configs to match new structure"""
    replacement_rules: dict = {
        "mappings": "submodules",
        "solvers": "submodules",
        "devices": "submodules"
    }
    if isinstance(obj, dict):
        for key in [k for k in obj]:
            obj[key] = recursive_app_modules_convert(
                obj[key])
            if key == "devices":
                for device in obj["devices"]:
                    if not "args" in device:
                        device['args'] = {}
                    device['args']["device_name"] = device['name']

            if key in replacement_rules:
                logging.warning(
                    "Found %s in app-modules -> replace with %s",
                    key, str(replacement_rules[key])
                )

                obj[replacement_rules[key]] = obj.pop(key)

    elif isinstance(obj, list):
        for i, element in enumerate(obj):
            obj[i] = recursive_app_modules_convert(element)
    elif isinstance(obj, str) and obj in replacement_rules:
        logging.warning("Found %s in app-modules -> replace with %s",
                        obj, replacement_rules[obj]
                        )
        return replacement_rules[obj]
    return obj


class ConfigTransformer():
    """class to convert the config.yml on the fly from QUARK 1.0
    structure to QUARK 2.0 structure"""

    def __init__(self, config) -> None:
        self.input_config = config

    def to_quark2(self):
        """method to return the QUARK 2.0 structure of a given config"""
        if not "mapping" in self.input_config:
            return self.input_config

        def dict_to_list(old_structure_dict: dict) -> list:
            """draws the key of the dict 
            inside the element.name 
            and makes a list"""
            for key in old_structure_dict:
                old_structure_dict[key]['name'] = key
            return list(old_structure_dict.values())

        quark_config = deepcopy(self.input_config)
        quark_config['application']['submodules'] = dict_to_list(
            quark_config.pop('mapping'))
        for mapping in quark_config['application']['submodules']:
            mapping['submodules'] = mapping.pop("solver")
            for solver in mapping['submodules']:
                solver['submodules'] = solver.pop("device")
                for device in solver['submodules']:
                    device["submodules"] = []
        return quark_config


# if __name__ == "__main__":
#     #test:
#     cvr_initial = CodeValidationRun(
#         'QUARK-QLM/test_configs/application_tests_myQLM/app-modules-application_tests.json',
#         'QUARK-QLM/test_configs/application_tests_myQLM/siemens_no_circuit_depth.yml'
#         )

#     cvr_target = CodeValidationRun(
#         'QUARK-QLM/test_configs/application_tests_myQLM/app-modules-application_tests.json',
#         'QUARK-QLM/test_configs/application_tests_myQLM/'
#         'siemens_no_circuit_depth_quark2manuelmigration.yml'
#         )

#     prepared_config = ConfigTransformer(cvr_initial.config).to_quark2()

#     print(
#         '-'*30,'\n',"Soll\n",
#         json.dumps(cvr_target.config,sort_keys=True,indent=2))
#     print(
#         '-'*30,'\n',"Ist\n",
#         json.dumps(prepared_config,sort_keys=True,indent=2))

#     with open('soll.txt','w') as bla:
#         bla.write(json.dumps(cvr_target.config,sort_keys=True,indent=2))
#     with open('ist.txt','w') as bla:
#         bla.write(json.dumps(prepared_config,sort_keys=True,indent=2))
