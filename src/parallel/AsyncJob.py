


from abc import abstractmethod
import datetime
from enum import Enum
import logging
import pickle
import random
import time


from qat.qlmaas.result import AsyncResult as QaptivaAsyncResult



class AsyncStatus(Enum):
    UNKNOWN = 0
    SUBMITTED = 1
    DONE = 2
    FAILED = 3



class AsyncJobData():
    def __init__(self, module_name, raw_input_data: any = None) -> None:
        self._status: AsyncStatus = 0
        self._job_input_data = raw_input_data
        self._job_return_data = None
        self._module_name = module_name
        self._job_info: dict = {}
        self.metrics = {}
        
        
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
    @abstractmethod
    def status(self):
        """The status of the job on the server. In this POC case, the status turns to DONE after  0.001s. 
        If the status comes from the QLM, then the Connection().get_info(id=job_info["id"]) would be my 
        idea to resolve the status. this property might be overwritten in a QLM specific child class 
        """
        return self._status

    #@abstractmethod
    @property
    def result(self) -> dict:
        """ returns the result of the raw input job as dict object"""
        if self.status == AsyncStatus.DONE:
            if isinstance(self._job_return_data, QaptivaAsyncResult): 
                return self._job_return_data.get_result()
        else:
            return None
    
    @abstractmethod
    def submit_to(self, stack):
        """ stack is a device like object that has the method submit(job: JOB), where 
        JOB is the raw input to the async job object"""
        self.job_info = stack.submit(self.input)
        self._job_info.update({
            "timestamp": datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
            })
        self._status = AsyncStatus.SUBMITTED
        
        
    @property
    def job_info(self) -> dict:
        """ stores the direct return value of the submission to the server, i.e.
        job_info = qpu.submit(...)
        e.g. the info such as as job_id, owner etc as assigned from QLM,
        see also felix' code. This ought to be json serializable"""
        return self._job_info
   
    @job_info.setter
    def job_info(self, value):
        if self._job_info:
            logging.error("job_info is tried to be overwritten")
            return
        self._job_info_type = value.__class__.__name__
        self._job_info = {}
        self._job_info['server_response'] = value
        
    
    def _primitivize(self, value):
        # print(f"primitivize {value} of type {type(value)}")
        if isinstance(value,(bool, str, int, float, type(None))):
            return value
        if isinstance(value,list):
            return list([self._primitivize(element) for element in value if self._primitivize(element) is not None])
        if isinstance(value,tuple):
            return tuple([self._primitivize(element) for element in value if self._primitivize(element) is not None])
        if isinstance(value, dict):
            return dict({key:self._primitivize(element) for key,element in value.items() if self._primitivize(element) is not None})
        return f"{type(value)}-object"
        
    def set_info(self,**kwargs):
        """method to manually set job_info."""
        self._job_info.update(kwargs)
    
    @abstractmethod    
    def collect_info_from_server(self) -> dict:
        """
        returns the original ResultInfo object from server
        //returns a dict representation of _job_info.
        //integrate the Qaptiva:AsyncResult.get_info() in case of a real async job"""
        return {}
    
    def get_json_serializable_info(self):
        return self._primitivize(self.job_info)

class AsyncPOCJobData(AsyncJobData):
    @property
    def status(self):# TODO status as parameter in dummy module
        """The status of the job on the server. In this POC case, the status turns to DONE after  0.001s. 
        If the status comes from the QLM, then the Connection().get_info(id=job_info["id"]) would be my 
        idea to resolve the status. this property might be overwritten in a QLM specific child class 
        """
        return AsyncStatus.DONE
    
    def submit_to(self, stack):
        """ stack is a device like object that has the method submit(job: JOB), where 
        JOB is the raw input to the async job object"""
        
        self.job_info = stack.submit(self.input) #calls a @property setter
        self._job_info.update({ 
            "id": int(random.random()*1000),
            "owner": "Smasch",
            "submission_time": time.time(),
            "timestamp": datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
            })
        self._status = AsyncStatus.SUBMITTED
        #dummy submission:
        dummy_server_mock= f"dummyServer/job{self._job_info['id']}.pkl"
        with open(dummy_server_mock, 'wb') as mock_file:
            pickle.dump(self.job_info, mock_file)
            logging.info("file %s written",dummy_server_mock )
        
    def collect_info_from_server(self) -> dict:
        dummy_server_mock= f"dummyServer/job{self._job_info['id']}.pkl"
        with open(dummy_server_mock, 'rb') as mock_file:
            self._job_info = pickle.load(mock_file)
            logging.info("file %s loaded", dummy_server_mock )
        return self.job_info
    
    @property
    def result(self) -> dict:
        if self.status == AsyncStatus.DONE:
            return self.job_info['server_response']
        else:
            return None

class QaptivaAsyncJob(AsyncJobData):
    pass

