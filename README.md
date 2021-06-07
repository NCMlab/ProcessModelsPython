# ProcessModelsPython

## MakeScripts.py
This defines all of the parameter values to cycle over. It also writes out the scripts to submit to the the cluster queue.
At the same time it creates a submission list CSV file to keep track of all of the simulations it is running.


## OrganizeResults.py
This reads through all of the simulations results files and organizes them into a single table.


## SubmissionHandler.py
This script checks on teh cluster queue and resubmits jobs as needed. 

### CheckSubisssions()
This reads the submission list CSV file and checks all of the results files. If there are results available it updates the submision list CSV file.

### ResubmitJobs()
If a result file is not available for a simulation, the batch job is submitted to the queue again.

### ProcessTools()
This should take as input: NBoot, NPower, ParameterList (betas, sample size), Model, OutDirectory

I can use the BatchArray input for the sample sizes.
I need to define in teh code an array and the SBATCH Array will provide an index to that array.