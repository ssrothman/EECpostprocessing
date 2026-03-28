universe                  = vanilla
executable                = WORKINGDIR/condor_exec.sh
arguments                 = $(Process)
request_memory            = 4gb
request_cpus              = 1
should_transfer_files     = NO
output                    = WORKINGDIR/condor/NAME_$(ClusterId)_$(ProcId).out
error                     = WORKINGDIR/condor/NAME_$(ClusterId)_$(ProcId).err
log                       = WORKINGDIR/condor/NAME_$(ClusterId)_$(ProcId).log
max_retries               = 1
+JobFlavour               = "workday"

queue NJOBS
