# SALBP - 1
As a first simple dataset for assembly line balancing, we use the [Benchmark Data Sets by Otto et al. (2013)](https://assembly-line-balancing.de/salbp/benchmark-data-sets-2013/)


## General Parameters Explanation

### \<number of tasks>
Total number of tasks to be assigned to stations.

### \<cycle time>
Time that is available per station.

### \<order strength>
Indicates the degree of connection between the tasks; 
Calculated by dividing the number of direct and indirect (transitive) 
connections by the number of possible connections.

### \<task times>
List of tasks with needed time to complete them. 

### \<precedence relations>
Restrictions on the order in which tasks are performed. 
If (task i, task j) is in the list, then task j must not be planned before task i. 
Only direct precedence relations should be included.
