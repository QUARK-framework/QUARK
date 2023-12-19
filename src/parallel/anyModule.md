


# Class diagram

```mermaid
classDiagram
DigitalDevice <|-- AsyncDevice
AsyncCore <|-- AsyncDevice
AsyncCore <|-- AsyncSolver
SomeSolverWithQAOALogic <|-- AsyncSolver
AsyncJob <|-- AsyncPOCJob
AsyncJob <|-- AsyncQaptivaJob
note for AsyncSolver "Solver with access to a device"
AsyncCore: +bool is_asynchron
class AsyncCore{
    +preprocess()
    +postprocess()
    +abstract submit()
    +abstract collect()
    +abstract sequencial_run()
}
class DigitalDevice{
    getQPU()
}
class SomeSolverWithQAOALogic{
    device_wrapper
}
class AsyncSolver{
    submit()
    collect()
    sequencial_run()
}
class AsyncDevice{
    submit()
    collect()
    sequencial_run()
}
class SomeSolverWithQUARKLogic{
    sequencial_run()
}
class AsyncJob{
    property result
    property status
    abstract get_result()
    abstract get_status()
}
```

# Sequence diagramgs

## Sequence: Async in leaf or postprocess

```mermaid
sequenceDiagram
participant BM as BenchmarkMgr
participant MA as Mapping(Core)
participant SO as Solver(Core)
participant AD as AsyncDevice(AsyncCore)
BM ->> +BM: sequencial_run_benchmark()
BM ->> +BM:traverse_config(preprocessed_input: dict )
BM ->> +MA: preprocess(problem: dict)
MA -->> -BM: return mapped_problem: dict
BM ->> +BM:traverse_config(preprocessed_input:  dict)
BM ->> +SO: preprocess(mapped_problem: dict {Circuit})
SO -->> -BM: return job: Job
BM ->> +BM:traverse_config(preprocessed_input: Job|JobWrapper)
BM ->> +AD: preprocess/postprocess/sequencial_run(job: Job|JobWrapper)
AD -->> -BM: return async_job: JobResult | JobWrapper
BM -->> -BM:traverse_config return 
opt not isinstance(async_job,JobResult)
BM ->> +SO: postprocess(job_result: JobResult)
SO -->> -BM: return solution: problem_solution
end
BM -->> -BM:traverse_config return 
opt not isinstance(async_job,JobResult)
BM ->> +MA: postprocess(problem: dict)
MA -->> -BM: return job: Job|JobWrapper
end
BM -->> -BM: traverse_config return 
```



## Sequence: Async happens in preprocess
```mermaid
sequenceDiagram
participant BM as BenchmarkMgr
participant MA as Mapping(Core)
participant PRE as LongPreprocessingModule(AsyncCore)
participant SO as Solver(Core)
participant AD as AsyncDevice(Core)
BM ->> +BM: sequencial_run_benchmark()
BM ->> +BM:traverse_config(preprocessed_input: dict )
BM ->> +MA: preprocess(problem: dict)
MA -->> -BM: return job: Job|JobWrapper
BM ->> +BM:traverse_config(preprocessed_input: dict )
BM ->> +PRE: preprocess(problem: dict)
PRE -->> -BM: return job: Job|JobWrapper
Note left of BM: if opt boxes is not executed parse innput as output
opt isinstance(async_job,JobResult)
BM ->> +BM:traverse_config(preprocessed_input:  dict)
BM ->> +SO: preprocess(problem: dict)
SO -->> -BM: return job: Job|JobWrapper
BM ->> +BM:traverse_config(preprocessed_input: Job|JobWrapper)
BM ->> +AD: preprocess/postprocess/sequencial_run(job: Job|JobWrapper)
AD -->> -BM: return async_job: JobResult | JobWrapper
BM -->> -BM:traverse_config return 
BM ->> +SO: postprocess(job_result: JobResult)
SO -->> -BM: return solution: problem_solution
BM -->> -BM:traverse_config return 
BM ->> +PRE: postprocess(problem: dict)
PRE -->> -BM: return job: Job|JobWrapper 
end
BM -->> -BM:traverse_config return 
opt not isinstance(async_job,JobResult)
BM ->> +MA: postprocess(problem: dict)
MA -->> -BM: return job: Job|JobWrapper
end
BM -->> -BM: traverse_config return 
```

# Messi all-modules-are-Async diagram 

It is also possible with the current state of the BenchmarkManager to have several/all modules Async.

TODO: draw diagram




# Any pre- or postprocessing module has the common form:

```mermaid
sequenceDiagram
participant BM as BenchmarkManager
participant any as AnyModule
participant AJob as AsyncJobData
Note over BM: input_job: the argument of the Xprocess
Note over any: output_result: the return value of the synchronous Xprocess | AsyncJobData object
Note over AJob,Connection: AsyncJobData knows about how to retrieve info from the server
BM ->> BM: get previous_job_info from old results.json
Note over BM: previous_job_info: simple element dictionary containing e.g. job id, timestamt,... etc
BM ->> +any: Xprocess(input_job: InputJob, previous_job_info: dict)
activate AJob
Note over any: Xprocess: either pre- or postprocess

alt collect mode
    Note right of any: Collect mode is determined by argument previous_job_info
    any ->> +any: collect()
    Note over any: Collect data based on info stored in previous_job_info
    any ->> AJob: AsyncJobData.status
    AJob ->> +Connection: get_connection
    AJob ->> Connection: get_job_status(AsyncJobData.job_info["id"])
    Connection -->> AJob: job status
    AJob -->> any:  status

    loop while status is pending | KeyboardInterupt
        any ->> AJob: AsyncJobData.status
        AJob -->> any:  status
    end
    any ->> AJob: AsyncJobData.result

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

    AJob -->> -any:  output_result: OutputResult | JobWrapper
    any-->> -any:  output_result

else submit mode
    Note right of any: submit mode has empty previous_job_info {} or None
    any ->> +any: submit()
    Note over any: Submitting to server
    any ->> +AJob: __init__
    AJob -->> any : async_job: AsyncJobData
    any ->> +Connection: submit (e.g.  AnyModule.getQPU().submit(job))
    Connection -->> -any: server_response: (Qaptiva)AsyncJob
    any-->> AJob: async_job.job_info.append(server_response : (Qaptiva)AsyncJob) [extract relevant fields]
    any -->> -any: return job_data
    any -->> any: AnyModule.metrics.add_metric("job_info", job_info)
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

any -->> -BM: Xprocess() -> OutputType | AsyncJobData
alt is AsyncJobData
BM -->> BM: come to an end
else
BM -->> BM: continue with retrieved data
end
```