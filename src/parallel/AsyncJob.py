


from abc import abstractmethod
from enum import Enum
import logging
import time


from qat.qlmaas.result import AsyncResult as QaptivaAsyncResult


def unpack_async_job_if_possible(data, module, stage):
    if isinstance(data, AsyncJobData):
        return data.unpack_if(module=module, stage=stage)
    else:
        return data

class AsyncStatus(Enum):
    UNKNOWN = 0
    SUBMITTED = 1
    DONE = 2
    FAILED = 3

class AsyncDeviceMode(Enum):
    AUTO = 0
    SUBMIT = 1
    COLLECT = 2

class ModuleStage(Enum):
    pre = 1
    post = 2

class AsyncResult():
    def get_result(self):
        return "hello"




class AsyncJobData():
    def __init__(self, module_name, module_stage, raw_input_data: any = None) -> None:
        self._status: AsyncStatus = 0
        self._job_input_data = raw_input_data
        self._job_return_data: AsyncResult = None
        self._module_name = module_name
        self._module_stage = module_stage # qualifies the stage of the module, i.e. 'pre' or "post" processing; technically this can be an arbitrary keyword, but the benchmarkManager accepts "pre" or "post"
        self._job_info = {} # eg the info as job_id, owner etc as assigned from QLM, see also felix' code. This ought to be json serializable
        
    @property
    def module_name(self):
        return self._module_name


    @property
    def input(self):
        return self._job_input_data

    @input.setter
    def input(self,value):
        self._job_input_data = value


    @property
    def status(self):
        try:
            submission_time = self.job_info["submission_time"]
            time_since = time.time()-submission_time
            print(time_since)
            if time_since>0.001:
                self._status = AsyncStatus.DONE
        except:
            self._status = AsyncStatus.FAILED
        return self._status

    @property
    def result(self):
        if self.status == AsyncStatus.DONE:
            if isinstance(self._job_return_data, QaptivaAsyncResult):
                return self._job_return_data.get_result()
            else:
                return "myDummy"
        else:
            return None

    @property
    def job_info(self) -> dict:
        return self._job_info
    
    def set_info(self,**kwargs):
        self._job_info = kwargs
        
    def get_info(self) -> dict:
        return self.job_info

    def unpack_if(self, **conditions):
        if self.status.name == "DONE":
            if self.module_name == conditions.get("module", self.module_name) and \
                self._module_stage.name == conditions.get("stage", self._module_stage.name):
                    # if conditions are set, only unpack 
                print(f"unpacking {self}")
                return self.result
            else:
                logging.info(f"Job {self.module_name} is already done")
                return self
        else:
            return self
                




class QaptivaAsyncJob(AsyncJobData):
    def __init__(self, raw_input_data: any = None) -> None:
        super().__init__(raw_input_data)
        self.async_result: QaptivaAsyncResult

