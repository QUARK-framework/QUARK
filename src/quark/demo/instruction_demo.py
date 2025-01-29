import logging

from quark.BenchmarkManager import Instruction
from quark.modules.Core import Core
from quark.modules.applications.Application import Application


class InstructionDemo(Application):
    """
    A simple QUARK Application implementation showing the usage of instructions.
    """

    def __init__(self, application_name: str = None):
        super().__init__(application_name)
        self.submodule_options = ["Dummy"]

    def preprocess(self, input_data: any, config: dict, **kwargs) -> tuple:
        """
        Preprocess input data with given configuration and instructions.

        :param input_data: Data to be  processed.
        :param config: Configuration for processing the data
        :param kwargs: Additional keyword arguments
        :return: Instruction, processed data, and processing time.
        """
        logging.info("%s", kwargs.keys())
        logging.info("previous_job_info: %s", kwargs.get("previous_job_info"))

        rep_count = kwargs["rep_count"]
        instruction_name = config.get("instruction", Instruction.PROCEED.name)
        instruction = Instruction.PROCEED

        if instruction_name == Instruction.PROCEED.name:
            instruction = Instruction.PROCEED
        elif instruction_name == Instruction.INTERRUPT.name:
            instruction = Instruction.INTERRUPT
        if instruction_name == "mixed":
            instruction = Instruction.PROCEED
            if rep_count % 2 == 1:
                instruction = Instruction.INTERRUPT
        elif instruction_name == "exception":
            raise Exception("demo exception")

        logging.info(
            "InstructionDemo iteration %s returns instruction %s",
            rep_count, instruction.name
        )
        return instruction, "", 0.

    def get_parameter_options(self) -> dict:
        """
        Returns parameter options for the preprocess method.
        """
        return {
            "instruction": {
                "values": [
                    Instruction.PROCEED.name,
                    Instruction.INTERRUPT.name,
                    "exception",
                    "mixed"
                ],
                "description": "How should preprocess behave?"
            }
        }

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule for the given option.

        :param option: The submodule option
        :return: Default submodule
        """
        return Dummy()

    def save(self, path: str, iter_count: int) -> None:
        """
        Saves the current state to the specified path.

        :param path: Path where the state should be saved
        :param iter_count: Iteration count.
        """
        pass


class Dummy(Core):
    """
    Dummy QUARK module implementation which is used by the InstructionDemo.
    """

    def get_parameter_options(self) -> dict:
        """
        Returns parameter options for the Dummy module.

        :return: Dictionary containing parameter options
        """
        return {}

    def get_default_submodule(self, option: str) -> Core:
        """
        Returns the default submodule for the given option.

        :param option: The submodule option
        """
        pass
