from abc import ABC, abstractmethod
import logging

from modules.Core import Core
from parallel.AsyncJob import AsyncJobData, AsyncStatus


class AsyncCore(Core, ABC):

    def get_parameter_options(self) -> dict:
        return {"async":
                    {"values": [True, False]}
                }

    def preprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        asynchronous_job_info = kwargs.get("asyncrounous_job_info", dict())
        prev_run_job_info = None if not asynchronous_job_info else asynchronous_job_info.get("job_info", False)
        if prev_run_job_info:
            logging.info("Resuming previous run with job_info = %s", prev_run_job_info)
            rv, result = self.collect(prev_run_job_info)
            if rv:
                return result, 0.0 # timeer?
            else:
                async_job_data = ???
                return async_job_data, 0.0
        elif self.config["async"]:
            job_info = self.submit(input_data, config, **kwargs)
            async_job_data = AsyncJobData() # async_job_data contains timestamp
            async_job_data.job_info = job_info
            self._status = AsyncStatus.SUBMITTED

            self._store_job_info(job_info)
            return async_job_data, 0.0
        else:
            return self.run(input_data, config, **kwargs)

    @abstractmethod
    def run(self, input_data: any, config: dict, **kwargs) -> any:
        pass

    @abstractmethod
    def submit_async(self, input_data: any, config: dict, **kwargs) -> any:
        """
        return value must be a json representable object - will be stored and
        passed as collect parameter in the QUARK resume run.
        """
        pass

    @abstractmethod
    def collect(self, job_info):
        """
        job_info: the object that was returned by submit on the previous run
        TODO: return value must contain the result or something that indicates that user has to try later again.
        """
        pass

    def _store_job_info(self, job_info: dict):
        """Storing the information about the submitted jobs to file"""
        print(f'storing job info: {job_info}')
        self.metrics.add_metric("job_info", job_info)
