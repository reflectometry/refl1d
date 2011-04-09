
import os
import datetime
from random import random

from sqlalchemy import or_

from . import runjob, jobid, store, db
from .notify import notify


class Scheduler(object):
    def __init__(self):
        db.connect()
        self.session = db.Session()

    def jobs(self, status=None):
        with self._lock:
            if status is None:
                result = self._jobs[:]
            else:
                result = [j for j in self._jobs if self._status[j] == status]
        return result
    def submit(self, request, origin):
        n = self.session.query(db.Job) \
            .filter(or_(notify=request['notify'],origin=origin)) \
            .filter(db.Job.date < datetime.utctime() - datetime.timedelta(30)) \
            .count()
        job = db.Job(name=request['name'],
                     notify=request['notify'],
                     date=datetime.utcnow(),
                     origin=origin,
                     priority=n+random())
        self.session.add(job)
        self.session.commit(job)
        store.create(job.id)
        store.put(id,'request',request)
        return job.id

    def _getjob(self, id):
        return self.session.query(db.Job).filter_by(id=id).first()

    def results(self, id):
        try:
            return runjob.results(id)
        except KeyError:
            return { 'status': 'UNKNOWN' }

    def status(self, id):
        job = self._getjob(id)
        return job.status

    def info(self,id):
        request = store.get(id,'request')
        request['id'] = id
        return request

    def cancel(self, id):
        self.session.query(db.Job) \
             .filter_by(id=id) \
             .filter(or_(db.Job.status=='ACTIVE', db.Job.status=='PENDING')) \
             .update({ 'status': 'CANCEL' })
        session.commit()

    def delete(self, id):
        """
        Delete any external storage associated with the job id.  Mark the
        job as deleted.
        """
        self.session.query(db.Job).filter_by(id=id).delete()
        store.destroy(id)

    def nextjob(self, request):
        # Make the next PENDING job active, where pending jobs are sorted
        # by priority.  Priority has a random value to ensure fair sharing
        # amongst users
        job = self.session.query(db.Job) \
            .filter_by(status='PENDING') \
            .order_by(db.Job.priority) \
            .limit(1) \
            .update({'status': 'ACTIVE',
                     'start': datetime.utcnow(),
                     }) \
            .one()
        activejob = db.ActiveJob(job=job,
                                 queue=request['queue'])
        request = store.get(id,'request')
        request['id'] = id
        notify(user=job.notify,
               msg=("Job %s started on %s at %s"
                    % (job.name,request['queue'],job.start)),
               level=1)
        return request

    def postjob(self, id, result):
        # TODO: redundancy check,
        job = self.session.query(db.Job) \
            .filter_by(id=id) \
            .update({'status': result['status'],
                     'stop': datetime.utcnow(),
                     })
        session.commit()
        store.put(id,'result',result)
        notify(user=job.notify,
               msg=("Job %s status=%s at %s"
                    % (job.name,job.status,job.stop)),
               level=2)
