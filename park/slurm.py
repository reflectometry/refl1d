import os
import subprocess

from park import config
from park import environment

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
        write_slurmid(jobid,slurmid)

    def status(self, jobid, workdir):
        return "UNKNOWN"

    def cancel(self, jobid):
        #print "canceling",jobid
        slurmid = read_slurmid(jobid)
        _slurm_exec('scancel',slurmid)

def read_slurmid(jobid):
    fid = open(os.path.join(config.storagedir(),jobid,'slurmid'), 'r')
    slurmid = fid.read()
    fid.close()
    return slurmid

def write_slurmid(jobid,slurmid):
    fid = open(os.path.join(config.storagedir(),jobid,'slurmid'), 'w')
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
