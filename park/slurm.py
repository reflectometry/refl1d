import os
import subprocess

from park import config
from park import environment

# Queue status words
_ACTIVE = ["RUNNING", "COMPLETING"]
_INACTIVE = ["PENDING", "SUSPENDED"]
_ERROR = ["CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL"]
_COMPLETE = ["COMPLETED"]


class Scheduler(object):
    def jobs(self):
        """
        Return a list of jobs on the queue.
        """
        #TODO: list completed but not deactivated as well as completed
        #print "queue"
        output,_ = _slurm_exec('squeue','-o', '%i %M %j')
        #print "output",output
        return output
    
    def deactivate(self, jobid):
        #TODO: remove the job from the active list
        pass

    # Job specific commands
    def queue_service(self, jobid, service, kernel, jobdir):
        """
        Put a command on batch queue, returning its job id.
        """
        #print "submitting job",jobid
        num_workers = config.tasks()

        script = os.path.join(jobdir,"J"+jobid)
        commands = ['export %s="%s"'%(k,v) for k,v in config.env().items()]
        commands += ["srun -n 1 nice %s >service.out&"%(service,),
                     "srun -n %d nice %s"%(num_workers,kernel)]
        create_batchfile(script,commands)

        _,err = _slurm_exec('sbatch', '-n',str(num_workers),
                            '-o', 'output',
                            '-D',jobdir,script)
        if not err.startswith('sbatch: Submitted batch job '):
            raise RuntimeError(err)
        slurmid = err[28:].strip()
        write_slurmid(jobdir,slurmid)

    def status(self, jobid, jobdir):
        """
        Returns the follow states:
        PENDING   --  Job is waiting to be processed
        ACTIVE    --  Job is busy being processed through the queue
        COMPLETE  --  Job has completed successfully
        ERROR     --  Job has either been canceled by the user or an 
                      error has been raised
        """

        state = ''
        inqueue = False
        hasresult = _find_result(jobdir)
        slurmid = read_slurmid(jobdir)

        out,_ = _slurm_exec('squeue', '-h', '--format=%i %T')
        out = out.strip()
        if out != "":
            for line in out.split('\n'):
                line = line.split()
                if slurmid == line[0]:
                    state = line[1]
                    inqueue = True
                    break

        if inqueue:
            if state in _ACTIVE:
                return "ACTIVE"
            elif state in _INACTIVE:
                return "PENDING"
            elif state in _COMPLETE:
                return "COMPLETE"
            elif state in _ERROR:
                return "ERROR"
            else:
                raise RuntimeError("unexpected state from squeue: %s"%state)
        else:
            if hasresult:
                return "COMPLETE"
            else:
                return "ERROR"

    def cancel(self, jobid, jobdir):
        #print "canceling",jobid
        slurmid = read_slurmid(jobdir)
        _slurm_exec('scancel',slurmid)

def _find_result(dir):
    import os
    pth = os.path.join(dir, 'result')
    return os.path.isfile(pth)

def read_slurmid(jobdir):
    fid = open(os.path.join(jobdir,'slurmid'), 'r')
    slurmid = fid.read()
    fid.close()
    return slurmid

def write_slurmid(jobdir,slurmid):
    fid = open(os.path.join(jobdir,'slurmid'), 'w')
    slurmid = fid.write(slurmid)
    fid.close()

def create_batchfile(script, commands):
    """
    Create the batchfile to run the job.
    """
    fid = open(script,'w')
    fid.write("#!/bin/sh\n")
    fid.write("\n".join(commands))
    fid.write("\nwait\n")
    fid.close()
    return script

def _slurm_exec(cmd, *args):
    """
    Run a slurm command, capturing any errors.
    """
    #print "cmd",cmd,"args",args
    process = subprocess.Popen([cmd]+list(args),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    out,err = process.communicate()
    if err.startswith(cmd+': error: '):
        raise RuntimeError(cmd+': '+err[15:].strip())
    return out,err
