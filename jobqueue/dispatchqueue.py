
import os
import datetime

from sqlalchemy import or_

from . import runjob, jobid, store, db


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
        job = db.Job(name=request['name'],
                     notify=request['notify'],
                     date=datetime.utcnow(),
                     origin=origin)
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
        self.session.query(db.Job).filter_by(id=id).delete()
        store.destroy(id)
