# Async mode
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

# Details
submit mode
```mermaid
sequenceDiagram
    actor main
    participant CM as ConfigMgr
    participant BM as BenchmarkMgr
    participant AD as AsyncDevice
    participant AJob as AsyncJob
    main ->> CM: add_output_directory(resume_dir)
    main ->> +BM: orchestrate_benchmark()
    BM ->> +BM: run_benchmark()
    Note over BM: load result from previous run
    BM ->> AD: preprocess(.., asynchronous_job_info)
    AD ->> AJob: init()
    Note over AD: store job meta info results
    AD -->> BM: AsyncJob
    BM -->> -BM: 
    BM -->> -main: 
    
    
```
resume mode
```mermaid
sequenceDiagram
    
    actor main
    participant CM as ConfigMgr
    participant BM as BenchmarkMgr
    participant AD as AsyncDevice
    main ->> CM: add_output_directory(resume_dir)
    main ->> +BM: orchestrate_benchmark()
    BM ->> +BM: run_benchmark()
    Note over BM: load job info from results.json
    BM ->> AD: preprocess(.., job_info)
    Note over AD: load result <br> typically from compute server
    AD -->> BM: result
    BM -->> -BM: 
    BM -->> -main: 

```