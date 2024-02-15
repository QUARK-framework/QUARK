# Async mode

For the Async mode two important classes are introduced:

- The __AsyncCore__, inheriting from Core, from which a QUARK2 module should inherit from to enable the async functionality
- The __AsyncJobManager__, which is a an abstract base class from which specific AsyncJobManager need to inherit from. The derived classes encode the site-specific logics


## The AsyncCore module

Instead of inheriting from Core the user needs to inherit from AsyncCore and call the super-constructor with the argument interruptable="PRE", "POST" or even "PREPOST" depending on which part of the module should potentially run asynchronous. Unlike a normal module, the user now needs to implement the methods submit_XXXprocess and collect_XXXprocess.
The submit_X is required, whereas if the collect_X is not implemented, the default implementation just passes the server result.
Note that no 
To facilitate the conversion of a conventional QUARK module such as a solver, it is also possible to inherit from AsyncCore and Solver or an already implemented class.
Within the class definition, the JobManager variable must be set to the desired AsyncJobManager class. 

```python

import MyJobManager
class MyAsyncUpgradedSolver(AsyncCore, SomeSpecificExistingModule):
    
    JobManager = MyJobManager
    
    def __init__(self, name: str = None):
        AsyncCore.__init__(self, interruptable="POST", name=name)
        SomeSpecificExistingModule.__init__(self)
    
    def submit_postprocess(self, input_data, config, **kwargs):
        ...        
        return stack.submit(job) # return the return value of the submision 
    
    def collect_postprocess(self, server_result):
        result = do_something_with_the_resultserver_result
        final_energy = result.value
        return final_energy, result
```
## The AsyncJobManager 

Within the AsyncJobManager the complete logic regarding the communication with a specific asynchronous process (e.g. QLMaaS, Qiskit-IBMQ) is handled. The QaptivaAsyncJobManager is already provided. 

```python
class MyJobManager(AsyncJobManager):
    def get_result(self) -> AsyncResult:
        """returns the object, that is returned by the Service (e.g. Qaptiva)"""
        return result
    
    def get_status(self) -> AsyncStatus:
        # returning  AsyncStatus.FAILED, AsyncStatus.DONE or AsyncStatus.SUBMITTED
        ...

    
    def _handle_submit_info(self, value: AsyncResult):
        # the input "value" is the object that is returned as response, when using the 
        # function that submits something to the server, e.g:
        # set all information that are crucial for identification of the job afterwards
        self.set_info(id=value.job_id)
        self.set_info(host=value.connection.hostname)
        self._job_info["server_response"] = value
```

## Class structure

```mermaid
classDiagram
SomeSpecificExistingModule <|-- MyAsyncUpgradedSolver
AsyncCore <|-- MyAsyncUpgradedSolver

AsyncJobManager <|-- MyJobManager
AsyncJobManager <|-- AsyncQaptivaJob
MyAsyncUpgradedSolver .. MyJobManager

class AsyncCore{
    +get_parameter_options(self)
    +preprocess() = if not hasattr(self,submit_preprocess,collect_preprocess) return super()
    +postprocess() = if not hasattr(self,submit_postprocess,collect_postprocess) return super()
    +sequencial_run() = collect(submit())  
}
class SomeSpecificExistingModule{
    getQPU()
}


class MyAsyncUpgradedSolver{
    JobManager = MyJobManager
    __init__() 
    submit_preprocess()
    collect_preprocess()
}
class AsyncJobManager{
    _handle_submit_info()
    abstract get_result()
    abstract get_status()
}
class MyJobManager{
    _handle_submit_info()
    get_result()
    get_status()
}
```

## Behind the scenes

The following is not absolutely necessary for the end user to know, but is intended to show how QUARK deals with asynchronous jobs.

### Instructions

Instruction is an enum and each XXXprocess return value prefixed with such an instruction so that the BenchmarkManager recognizes whether a continuation of the benchmark run is desired or whether simply the job-meta data should be saved in the results.json.

### Sequence Diagram


```mermaid
sequenceDiagram
participant BM as BenchmarkManager
participant any as MyAsyncUpgradedSolver
participant AJob as MyJobManager
Note over BM: input_job: the argument of the XXXprocess
Note over any: output_result: the return value of the synchronous XXXprocess | MyJobManager object
Note over AJob,Connection: MyJobManager knows about how to retrieve info from the server
BM ->> BM: get previous_job_info from old results.json
Note over BM: previous_job_info: simple element dictionary containing e.g. job id, timestamt,... etc
BM ->> +any: XXXprocess(input_job: InputJob, previous_job_info: dict)

Note over any: XXXprocess: either pre- or postprocess

any ->> +AJob: __init__
alt submit mode
    Note right of any: submit mode has empty previous_job_info {} or None
    any ->> +any: _submit() calls correct submit_XXXprocess
    Note over any: Submitting to server
    AJob -->> any : async_job: MyJobManager
    any ->> +Connection: submit (e.g.  AnyModule.getQPU().submit(job))
    Connection -->> -any: server_response: (Qaptiva)AsyncJob
    any-->> AJob: async_job.job_info.append(server_response : (Qaptiva)AsyncJob) [extract relevant fields]
    any -->> -any: return Instruction.INTERRUPT, job_info: MyJobManager
    any -->> any: AnyModule.metrics.add_metric("job_info", job_info)
    

else collect mode
    Note right of any: Collect mode is determined by argument previous_job_info
    any ->> +any: _collect()
    Note over any: Collect data based on info stored in previous_job_info
    any ->> AJob: MyJobManager.status
    AJob ->> +Connection: get_connection
    AJob ->> Connection: get_job_status(MyJobManager.job_info["id"])
    Connection -->> AJob: job status
    AJob -->> any:  status

    
    any ->> AJob: MyJobManager.result

    alt status is failed
        Note Over AJob: raise Exception
    else status is done
        AJob ->> Connection: get_job(id)
        Connection -->> AJob: job: (Qaptiva)AsyncJob
        deactivate Connection
        AJob->> AJob: return get_result()
    else is pending
        AJob ->> AJob: return self 
    end

    AJob -->> any:  output_result:( Instruction.PROCEED, OutputResult | Instruction.INTERRUPT,JobWrapper)
    any-->> any:  output_result
    deactivate AJob

else sequenzial mode
    any ->> +any: submit()
    any ->> +AJob: [same as above]
    AJob ->> +Connection: ...
    Connection ->> -AJob: ...
    AJob ->> -any: ...
    any -->> -any: submit()
    any ->> +any: collect()
    any ->> +AJob: ...
    AJob ->> +Connection: ...
    Connection ->> -AJob: ...
    AJob ->> -any: ...
    any -->> -any: collect()

end

any -->> -BM: XXXprocess() -> ( Instruction.PROCEED, OutputResult | Instruction.INTERRUPT,JobWrapper)
alt Instruction.INTERRUPTED
BM -->> BM: come to an end and safe meta_data
else
BM -->> BM: continue with retrieved data
end
```



# Basic Idea
The QUARK module M3 in the following graph supports asynchronous execution.
QUARK supplies a data type AsyncResult.
If the benchmark manager receives an AsyncResult from one of the modules it stops the execution of the current module 
stack and stores the result data written by the modules so far.
M3 writes all data needed to continue its task later.
```mermaid
flowchart LR
    M1 --> M2 --> M3
    --> B((AsyncResult))
```

continue-mode

The benchmark manager adds the
```mermaid
flowchart LR
    M1 --> M2 --> M3 --> M4
        PR((previous result)) --> M3
    M4 -->M3 --> M2 --> M1

```
