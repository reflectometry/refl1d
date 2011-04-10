
import os
from datetime import datetime, timedelta
from random import random

from sqlalchemy import and_, or_, not_, func, select, alias
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound

from . import runjob, jobid, store, db, notify
from db import Job, ActiveJob


class Scheduler(object):
    def __init__(self):
        db.connect()
        self.session = db.Session()

    def jobs(self, status=None):
        if status:
            jobs = (self.session.query(Job)
                .filter(Job.status==status)
                .order_by(Job.priority)
                )
        else:
            jobs = (self.session.query(Job)
                .order_by(Job.priority)
                )
        return [j.id for j in jobs]
    def submit(self, request, origin):
        # Find number of jobs for the user in the last 30 days
        n = (self.session.query(Job)
            .filter(or_(Job.notify==request['notify'],Job.origin==origin))
            .filter(Job.date >= datetime.utcnow() - timedelta(30))
            .count()
            )
        #print "N",n
        job = Job(name=request['name'],
                  notify=request['notify'],
                  origin=origin,
                  priority=n)
        self.session.add(job)
        self.session.commit()
        store.create(job.id)
        store.put(job.id,'request',request)
        return job.id

    def _getjob(self, id):
        return self.session.query(Job).filter(Job.id==id).first()

    def results(self, id):
        try:
            return runjob.results(id)
        except KeyError:
            return { 'status': 'UNKNOWN' }

    def status(self, id):
        job = self._getjob(id)
        return job.status if job else 'UNKNOWN'

    def info(self,id):
        request = store.get(id,'request')
        request['id'] = id
        return request

    def cancel(self, id):
        (self.session.query(Job)
             .filter(Job.id==id)
             .filter(Job.status.in_('ACTIVE','PENDING'))
             .update({ 'status': 'CANCEL' }) 
             )
        self.session.commit()

    def delete(self, id):
        """
        Delete any external storage associated with the job id.  Mark the
        job as deleted.
        """
        (self.session.query(Job)
             .filter(Job.id == id)
             .update({'status': 'DELETE'})
             )
        store.destroy(id)

    def nextjob(self, queue):
        """
        Make the next PENDING job active, where pending jobs are sorted
        by priority.  Priority is assigned on the basis of usage and the
        order of submissions.
        """

        # Define a query which returns the lowest job id of the pending jobs 
        # with the minimum priority
        _priority = select([func.min(Job.priority)],
                           Job.status=='PENDING')
        min_id = select([func.min(Job.id)],
                        and_(Job.priority == _priority,
                             Job.status == 'PENDING'))

        for i in range(10): # Repeat if conflict over next job
            # Get the next job, if there is one
            try:
                job = self.session.query(Job).filter(Job.id==min_id).one()
                #print job.id, job.name, job.status, job.date, job.start, job.priority
            except NoResultFound:
                return None

            # Mark the job as active and record it in the active queue
            (self.session.query(Job)
             .filter(Job.id == job.id)
             .update({'status': 'ACTIVE',
                      'start': datetime.utcnow(),
                      }))
            activejob = db.ActiveJob(jobid=job.id, queue=queue)
            self.session.add(activejob)
            
            # If the job was already taken, roll back and try again.  The
            # first process to record the job in the active list wins, and
            # will change the job status from PENDING to ACTIVE.  Since the
            # job is no longer pending, the  so this
            # should not be an infinite loop.  Hopefully if the process
            # that is doing the transaction gets killed in the middle then
            # the database will be clever enough to roll back, otherwise
            # we will never get out of this loop.
            try:
                self.session.commit()
            except IntegrityError:
                self.session.rollback()
                continue
            break
        else:
            logging.critical('dispatch could not assign job %s'%job.id)
            raise IOError('dispatch could not assign job %s'%job.id)

        request = store.get(job.id,'request')
        request['id'] = job.id
        notify.notify(user=job.notify,
                      msg=("Job %s started on %s at %s"
                           % (job.name,queue,job.start)),
                      level=1)
        return request

    def postjob(self, id, result):
        # TODO: redundancy check,
        (self.session.query(Job)
            .filter(Job.id == id)
            .update({'status': result['status'],
                     'stop': datetime.utcnow(),
                     })
            )
        (self.session.query(ActiveJob)
            .filter(ActiveJob.jobid == id)
            .delete())
        self.session.commit()
        store.put(id,'result',result)
        job = self._getjob(id)
        notify.notify(user=job.notify,
                      msg=("Job %s status=%s at %s"
                           % (job.name,job.status,job.stop)),
                      level=2)
