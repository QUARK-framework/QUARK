from abc import ABC, abstractmethod
from enum import Enum
import logging
import time




from modules.Core import Core
from parallel.AsyncJob import AsyncJobManager, AsyncStatus

class ModuleStage(Enum):
    none = 0
    pre = 1
    post = 2
    leaf = 3
    both = 4
    
    

class AsyncCore(Core, ABC):

    def get_parameter_options(self) -> dict:
        return {"async":
                    {"values": ["pre","post","both"]}
                }
                
    def preprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        return self._process(ModuleStage.pre, input_data, config, **kwargs)
    
    def postprocess(self, input_data: any, config: dict, **kwargs) -> (any, float):
        return self._process(ModuleStage.post,input_data, config, **kwargs)
        
    def _process(self, stage: ModuleStage, input_data: any, config: dict, **kwargs) -> (any, float): 
        """ Input data is the job
        returns the AsyncJobManager or the result
        """
        
        # check if *process is configured to run asynchron, else fallback to Core
        async_mode = config.get("async", "none")
        if not async_mode != "both" and async_mode != stage.name:
            if stage == ModuleStage.pre:
                return super().preprocess(input_data, config, **kwargs)
            if stage == ModuleStage.post:
                return super().postprocess(input_data, config, **kwargs)
            
        
        
        asynchronous_job_info =  kwargs.get("asynchronous_job_info", dict())
        synchron_mode = kwargs.get("synchron_mode",False)
        prev_run_job_info = None if not asynchronous_job_info else asynchronous_job_info.get("job_info", False)
        is_submit_job = not prev_run_job_info
        is_collect_job = not is_submit_job or synchron_mode
        
        job_manager =  AsyncJobManager (self.name, input_data,
                                        config, **kwargs) #TODO make the class an option
        if is_submit_job:
            return self._submit(stage, job_manager)
            
       
        if is_collect_job:
            logging.info("Resuming previous run with job_info = %s", prev_run_job_info)
            job_manager.set_info( **prev_run_job_info)
            return self._collect(stage,job_manager)
            
    
    
    def _submit(self, stage, job_manager: AsyncJobManager):
        """calls the corresponding submit_pre or postprocess function with arguments
        filled from job_manager"""
        submit = self.submit_preprocess \
            if stage == ModuleStage.pre \
                else self.submit_postprocess
        job_manager.job_info = submit(job_manager.input, 
                                        job_manager.config, 
                                        **job_manager.kwargs)
        
        self.metrics.add_metric("job_info", 
                                job_manager.get_json_serializable_info())
        
        return job_manager, 0.0
    
    
    def _collect(self, stage, job_manager):
        """calls the corresponding collect_pre or postprocess function with arguments
        filled from job_manager"""
        collect = self.collect_preprocess \
            if stage == ModuleStage.pre \
                else self.collect_postprocess
        
        if job_manager.status == AsyncStatus.FAILED:
            raise Exception(f"job {job_manager} failed")
        
        try:
            while job_manager.status == AsyncStatus.SUBMITTED:
                time.sleep(1)
            if job_manager.status == AsyncStatus.DONE:
                logging.info(f"job {job_manager} done")
            if job_manager.status == AsyncStatus.FAILED:
                raise Exception(f"job {job_manager} failed")
        except KeyboardInterrupt:
            pass
        
        self.set_metrics_of(job_manager)
            
        return job_manager.result, job_manager.runtime
    
    def submit_preprocess(self, job, config, **kwargs):
        """interface: overwrite this method to a module specific submission.
        return value is supposed to be the answer of the server call when submitting 
        
        e.g.
        ```
        qpu = self._get_qpu_plugin()
        server_response = qpu.submit(job)
        return server_response
        ```
        """
        raise NotImplementedError
    
    def collect_preprocess(self):
        """interface: overwrite this method to a module specific collect-call"""
        raise NotImplementedError
    
    def submit_postprocess(self, job, config, **kwargs):
        raise NotImplementedError
    
    def collect_postprocess(self):
        raise NotImplementedError

     
      
    def set_metrics_of(self, job: AsyncJobManager): 
        """Parsing metrics from a done job into the Device metrics"""
        for metric_key, metric_value in job.metrics.items():
            self.metrics.add_metric(metric_key, metric_value)


class AsyncPOCDevice(AsyncCore):
    pass

class AsyncQaptivaDevice(AsyncCore):
    pass
