from modules.Core import Core
from modules.applications.Application import Application
from utils import quark_stop_watch
from BenchmarkManager import Instruction
from parallel.AsyncCore import AsyncCore
from parallel.AsyncJob import AsyncJobManager, AsyncStatus

import logging
"""
This package supplies some simple QUARK components for demonstrating and testing
asynchronous QUARK modules.
ApplForPoc -> {"x":5} -> AsyncPreprocessForPoc -> {"job_id": "7q9wr"} -> INTERRUPT

"""


class ApplForPoc(Application):

    def __init__(self):
        super().__init__("ApplForPoc")

    def get_parameter_options(self) -> dict:
        return {}

    def get_default_submodule(self, option: str) -> Core:
        pass


    @quark_stop_watch()
    def preprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        return {"x": 5}  # the problem instance


    @quark_stop_watch()
    def postprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        self.metrics.add_metric("post_input", input_data)
        return input_data

    def save(self, path: str, iter_count: int) -> None:
        pass


class MyJobManager(AsyncJobManager):
    def get_status(self) -> AsyncStatus:
        logging.info("job info: %s", self.job_info)
        # Here, typically, the current job status will be retrieved from the server
        # using the information available in self.job_info
        # For the POC we return SUBMITTED on the first collect
        # and DONE on each further collect.
        if self.job_info.get("count", 1) > 1:
            return AsyncStatus.DONE
        else:
            return AsyncStatus.SUBMITTED

    def get_result(self) -> dict:
        # Here typically the job result gets retrieved from the server.
        # For the POC we return the job_id as result
        job_id= self.job_info["server_response"]["job_id"]
        return {"result": job_id}


class AsyncPreprocessForPoc(AsyncCore):
    """
    This QUARK module is used in the test cases as an intermediate module (mapping)
    as well as top level module (application) and also as leaf module.
    """

    JobManager = MyJobManager

    def __init__(self):
        super().__init__(
            interruptable="PRE"
        )  # "PRE", "POST" oder "PREPOST" (searches for substring)

    
    def get_default_submodule(self, option: str) -> Core:
        pass

    def submit_preprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        self.metrics.add_metric("pre_input", input_data)
        # Here the input_data typically would be submitted to a server
        # This is mocked here by hard coding some fake server response:
        server_response = {"job_id": "7q9wr"}
        return server_response

    def collect_preprocess(self, server_result):
        if not isinstance(server_result, AsyncJobManager):
            self.metrics.add_metric("server_result", server_result)
        else:
            count = server_result.job_info.get("count", 1)
            server_result.job_info["count"] = count + 1
            self.metrics.add_metric("server_result", "not available")
        return server_result

    @quark_stop_watch()
    def postprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        self.metrics.add_metric("post_input", input_data)
        input_data_post = {}
        input_data_post.update(input_data)
        input_data_post.setdefault("post_processed_by", []).append("AsyncPreprocessForPoc")
        return input_data_post

    def save(self, path: str, iter_count: int) -> None:
        """Needed in case that this module is used as application"""
        pass


class AsyncPostprocessForPoc(AsyncCore):
    """
    This QUARK module is used in the test cases as an intermediate module (mapping)
    as well as top level module (application) and also as leaf module.
    """

    JobManager = MyJobManager
    def __init__(self):
        super().__init__(
            interruptable="POST"
        )  # "PRE", "POST" oder "PREPOST" (searches for substring)

    def get_default_submodule(self, option: str) -> Core:
        pass

    def submit_postprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        self.metrics.add_metric("post_input", input_data)
        # Here the input_data typically would be submitted to a server
        # This is mocked here by hard coding some fake server response:
        server_response = {"job_id": "7q9wr"}
        return server_response

    def collect_postprocess(self, server_result):
        # TODO input_data should be available here
        if not isinstance(server_result, AsyncJobManager):
            self.metrics.add_metric("server_result", server_result)
        else:
            count = server_result.job_info.setdefault("count", 1)
            server_result.job_info["count"] = count + 1
            self.metrics.add_metric("server_result", "not available")
        return server_result

    @quark_stop_watch()
    def preprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        self.metrics.add_metric("pre_input", input_data)
        input_data_post = {}
        if input_data:
            # is None in case this module is used as application
            input_data_post.update(input_data)
        input_data_post.setdefault("pre_processed_by", []).append("AsyncPostprocessForPoc")
        return input_data_post

    def save(self, path: str, iter_count: int) -> None:
        """Needed in case that this module is used as application"""
        pass


class LeafForPoc(Core):

    def get_parameter_options(self) -> dict:
        return {}

    def get_default_submodule(self, option: str) -> Core:
        pass

    @quark_stop_watch()
    def preprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        self.metrics.add_metric("pre_input", input_data)
        return input_data

    @quark_stop_watch()
    def postprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        self.metrics.add_metric("post_input", input_data)
        input_data_post = {}
        input_data_post.update(input_data)
        input_data_post.setdefault("post_processed_by", []).append("LeafForPoc")
        return input_data_post

