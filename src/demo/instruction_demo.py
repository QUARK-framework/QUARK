import logging

from BenchmarkManager import Instruction
from modules.Core import Core
from modules.applications.Application import Application


class InstructionDemo(Application):
    """
    A simple QUARK Application implementation showing the usage of instructions.
    """
    def __init__(self, application_name: str = None):
        super().__init__(application_name)
        self.submodule_options = ["Dummy"]

    def preprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
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
            if rep_count%2 == 1:
                instruction = Instruction.INTERRUPT
        elif instruction_name == "exception":
            raise Exception("demo exception")

        logging.info("InstructionDemo iteration %s returns instruction %s", rep_count, instruction.name)
        return instruction, "", 0.

    def get_parameter_options(self) -> dict:
        return {
            "instruction": {"values": [Instruction.PROCEED.name,
                                       Instruction.INTERRUPT.name,
                                       "exception",
                                       "mixed"],
                            "description": "How should preprocess behave?"}
        }

    def get_default_submodule(self, option: str) -> Core:
        return Dummy()

    def save(self, path: str, iter_count: int) -> None:
        pass


class Dummy(Core):
    """
    Dummy QUARK module implementation which is used by the InstructionDemo.
    """

    def get_parameter_options(self) -> dict:
        return {}

    def get_default_submodule(self, option: str) -> Core:
        pass
