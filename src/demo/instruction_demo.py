import logging

from BenchmarkManager import Instruction
from modules.Core import Core
from modules.applications.Application import Application


class InstructionDemo(Application):
    def __init__(self, name: str = None):
        super().__init__(name)
        self.submodule_options = ["Dummy"]

    def preprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        logging.info("%s", kwargs.keys())
        logging.info("job_info: %s", kwargs.get("asynchronous_job_info"))
        rep_count = kwargs["rep_count"]
        instruction_name = config.get("instruction", Instruction.PROCEED.name)
        instruction = Instruction.PROCEED
        if instruction_name == Instruction.PROCEED.name:
            instruction = Instruction.PROCEED
        elif instruction_name == Instruction.SKIP.name:
            instruction = Instruction.SKIP
        elif instruction_name == Instruction.INTERRUPT.name:
            instruction = Instruction.INTERRUPT
        if instruction_name == "mixed":
            instruction = Instruction.PROCEED
            if rep_count%3 == 1:
                instruction = Instruction.SKIP
            elif rep_count%3 == 2:
                instruction = Instruction.INTERRUPT
        elif instruction_name == "exception":
            raise Exception("demo exception")

        logging.info("InstructionDemo iteration %s returns instruction %s", rep_count, instruction.name)
        return instruction, "", 0.

    def get_parameter_options(self) -> dict:
        return {
            "instruction": {"values": [Instruction.PROCEED.name,
                                       Instruction.SKIP.name,
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
    def get_parameter_options(self) -> dict:
        return {}

    def get_default_submodule(self, option: str) -> Core:
        pass
