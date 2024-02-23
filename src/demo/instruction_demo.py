from BenchmarkManager import Instruction
from modules.Core import Core
from modules.applications.Application import Application


class InstructionDemo(Application):

    def preprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        return Instruction.SKIP

    def get_parameter_options(self) -> dict:
        return {}

    def get_default_submodule(self, option: str) -> Core:
        return None

    def save(self, path: str, iter_count: int) -> None:
        pass

