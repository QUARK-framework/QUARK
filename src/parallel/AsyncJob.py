import os.path
from abc import ABC, abstractmethod
import datetime
from enum import Enum
import logging
import pickle
import random
import time


class AsyncModeException(Exception):
    pass


class AsyncStatus(Enum):
    UNKNOWN = 0
    SUBMITTED = 1
    DONE = 2
    FAILED = 3
    SYNCHRON = 4


class AsyncJobManager(ABC):
    """Base class for modules capable of performing asynchronous jobs.
    to use it for parallel computing of preprocess(postprocess) the
    method submit_preprocess(submit_postprocess) has to be overwritten
    optially, the collect_preprocess(collect_postprocess) can be overwritten
    to change

    """

    def __init__(
        self, module_name, input_data: any = None, config=None, **kwargs
    ) -> None:
        self._status = AsyncStatus.UNKNOWN
        self._job_input_data = input_data
        self._job_return_data = None
        self._module_name = module_name
        self._job_info: dict = {}
        self.metrics = {}
        self.config = config
        self.kwargs = kwargs
        self.server_connection = None

    def __repr__(self) -> str:
        return f"parallel job with ID={self.job_info.get('id') or 'UNKNOWN'}"

    @property
    def module_name(self):
        return self._module_name

    @property
    def input(self):
        return self._job_input_data

    @input.setter
    def input(self, value):
        self._job_input_data = value

    def status(self):
        """The status of the job on the server. In this POC case, the status turns to DONE after  0.001s.
        If the status comes from the QLM, then the Connection().get_info(id=job_info["id"]) would be my
        idea to resolve the status. this property might be overwritten in a QLM specific child class
        """
        return self._get_status()

    def _get_status(self):
        """memorizes if status was already FAILED or DONE before"""
        if self._status not in [AsyncStatus.DONE, AsyncStatus.FAILED]:
            self._status = self.get_status()
        return self._status

    @abstractmethod
    def get_status(self) -> AsyncStatus:
        """Server specific implementation of how to determine the job status"""

    def result(self):
        """returns the job result or self, if job is not finished"""
        status = self.status()
        assert isinstance(status, AsyncStatus), "Status is not of type AsyncStatus"
        if status == AsyncStatus.DONE:
            self.set_info(
                reception_time=datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
            )
            return self.get_result()
        if status == AsyncStatus.FAILED:
            self.set_info(
                reception_time=datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
            )
            try:
                self.get_result()
            except Exception as e:
                raise AsyncModeException from e
        return self

    @abstractmethod
    def get_result(self) -> dict:
        """returns the result of the raw input job as dict object"""

    @property
    def job_info(self) -> dict:
        """stores the direct return value of the submission to the server, i.e.
        job_info = qpu.submit(...)
        e.g. the info such as as job_id, owner etc as assigned from QLM,
        see also felix' code. This ought to be json serializable"""
        return self._job_info

    @job_info.setter
    def job_info(self, value):
        if self._job_info:
            logging.error("job_info is tried to be overwritten, rejecting")
            return
        self._job_info_type = value.__class__.__name__
        self._job_info = {
            "submission_time": datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        }
        self._status = AsyncStatus.SUBMITTED
        self._handle_submit_info(value)
        if "id" not in self.job_info:
            self.job_info["id"] = "unknown"
        # assert "id" in self.job_info, f"The job info does not contain 'id', please modify the"\
        #    f"{self.__class__.__name__}._handle_submit_info method and define a unique id"

    def _handle_submit_info(self, value):
        """
        simplest case is just to copy the value inside, or modify, if needed
        NOTE: self._job_info['id']  must be set now
        ```
        self._job_info['server_response'] = value
        assert 'id' in self._job_info, "'id' is needed"
        ```
        """

        self._job_info["server_response"] = value

    @property
    def runtime(self):
        """server site runtime in ms"""
        return self.job_info.get("runtime", 0.0)

    def _primitivize(self, value):
        # print(f"primitivize {value} of type {type(value)}")
        if isinstance(value, (bool, str, int, float, type(None))):
            return value
        if isinstance(value, Enum):
            return value.name
        if isinstance(value, list):
            return list(
                [
                    self._primitivize(element)
                    for element in value
                    if self._primitivize(element) is not None
                ]
            )
        if isinstance(value, tuple):
            return tuple(
                [
                    self._primitivize(element)
                    for element in value
                    if self._primitivize(element) is not None
                ]
            )
        if isinstance(value, dict):
            return dict(
                {
                    key: self._primitivize(element)
                    for key, element in value.items()
                    if self._primitivize(element) is not None
                }
            )
        return f"{type(value)}-object"

    def set_info(self, **kwargs):
        """method to manually set job_info."""
        self._job_info.update(kwargs)

    def get_json_serializable_info(self):
        """Should return a primitive object, that can be filled into the results.json
        and contain all necessary information to retrieve the job from server again
        and some bookkeeping"""
        return self._primitivize(self.job_info)


class POCJobManager(AsyncJobManager):
    def get_status(self):
        """The status of the job on the server. In this POC case, the status turns to DONE after  0.001s.
        If the status comes from the QLM, then the Connection().get_info(id=job_info["id"]) would be my
        idea to resolve the status. this property might be overwritten in a QLM specific child class
        """
        # runtime = time.time()-self.job_info.get("submission_time_s", 3000000000)
        # if runtime< 0.01:
        #    return AsyncStatus.SUBMITTED
        # if random.random() < 0.3:
        #     return AsyncStatus.FAILED
        return AsyncStatus.DONE

    def _handle_submit_info(self, value):
        """stack is a device like object that has the method submit(job: JOB), where
        JOB is the raw input to the async job object"""

        self._job_info["server_response"] = value

        self._job_info.update(
            {
                "server_response": value,
                "id": int(random.random() * 1000),
                "owner": "Smasch",
                "submission_time_s": time.time(),
            }
        )
        # dummy submission:
        dummy_server_mock = f"dummyServer/job{self._job_info['id']}.pkl"
        if not os.path.exists("dummyServer"):
            os.mkdir("dummyServer")
        with open(dummy_server_mock, "wb") as mock_file:
            pickle.dump(self.job_info, mock_file)
            logging.info("file %s written in %s", dummy_server_mock, os.getcwd())

    def get_result(self) -> dict:
        dummy_server_mock = f"dummyServer/job{self._job_info['id']}.pkl"
        with open(dummy_server_mock, "rb") as mock_file:
            self._job_info = pickle.load(mock_file)
            logging.info("file %s loaded", dummy_server_mock)
        return self.job_info.get("server_response", None)
