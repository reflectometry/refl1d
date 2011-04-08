import os
import cPickle as pickle
import flask
from flask import redirect, url_for, flash
from flask import send_from_directory
from werkzeug import secure_filename

from . import store

app = flask.Flask(__name__)

app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'secret'
app.config['SCHEDULER'] = 'slurm'
#app.config['SCHEDULER'] = 'direct'

def format_result(result, format='json', template=None):
    """
    Return result as a particular format.
    """
    #print "result",result
    if format == 'html':
        if template is None: flask.abort(400)
        return flask.render_template(template, **result)
    elif format == 'json':
        return flask.jsonify(**dict((str(k),v) for k,v in result.items()))
    elif format == 'pickle':
        return pickle.dumps(result)
    else:
        flask.abort(400) # Bad request


@app.route('/jobs.<format>', methods=['GET'])
def list_jobs(format='json'):
    """
    GET /jobs.<format>

    Return a list of all job ids.
    """
    result = dict(jobs=scheduler.jobs())
    return format_result(result, format, template='list_jobs.html')

@app.route('/jobs/<any(u"pending",u"active",u"error",u"complete"):status>.<format>',
           methods=['GET'])
def filter_jobs(status, format='json'):
    """
    GET /jobs/<pending|active|error|complete>.<format>

    Return all jobs with a particular status.
    """
    result = dict(jobs=scheduler.jobs(status=str(status).upper()))
    return format_result(result, format, template='list_jobs.html')


@app.route('/jobs.<format>', methods=['POST'])
def create_job(format='json'):
    """
    POST /jobs.<format>

    Schedule a new job, return the job record.
    """
    id = scheduler.submit(flask.request.json,
                          origin=flask.request.remote_addr)
    flash('Job %s scheduled' % id)
    result = scheduler.info(id)
    result['jobid'] = id
    #return redirect(url_for('show_job', id=id, format=format))
    return format_result(result, format=format, template='show_job.html')

@app.route('/jobs/<int:id>.<format>', methods=['GET'])
def show_job(id, format='json'):
    """
    GET /jobs/<id>.<format>

    Get job record by id.
    """
    result = scheduler.info(id)
    result['jobid'] = id
    return format_result(result, format=format, template='show_job.html')

@app.route('/jobs/<int:id>/result.<format>', methods=['GET'])
def get_results(id, format='json'):
    """
    GET /jobs/<id>/result.<format>

    Get job results by id.

    Returns { status: 'PENDING|ACTIVE|COMPLETE|ERROR|UNKNOWN' }
    If status is 'ERROR', then retval['trace'] is the traceback.
    If status is 'COMPLETE', then retval['result'] will contain the results
    of the job, which vary depending on the service requested.
    """
    result = scheduler.results(id)
    result['jobid'] = id
    #print "returning result",result
    return format_result(result, format=format)

@app.route('/jobs/<int:id>/status.<format>', methods=['GET'])
def get_status(id, format='json'):
    """
    GET /jobs/<id>/status.<format>

    Get job status by id.

    Returns { status: 'PENDING|ACTIVE|COMPLETE|ERROR|UNKNOWN' }
    """
    result = { 'status': scheduler.status(id) }
    result['jobid'] = id
    return format_result(result, format=format)


@app.route('/jobs/<int:id>.<format>', methods=['DELETE'])
def delete_job(id, format='json'):
    """
    DELETE /jobs/<id>.<format>

    Deletes a job, returning the list of remaining jobs as <format>
    """
    scheduler.delete(id)
    flash('Job %s deleted' % id)
    result = dict(jobs=scheduler.jobs())
    return format_result(result, format=format, template="list_jobs.html")
    #return redirect(url_for('list_jobs', id=id, format=format))

@app.route('/jobs/nextjob.<format>', methods=['POST'])
def fetch_work(format='json'):
    preference = flask.request.json
    request = sheduler.nextjob(preference)
    return format_result(result, format=format)

@app.route('/jobs/<int:id>/data/index.<format>')
def listfiles(id, format):
    try:
        path = store.path(id)
        files = sorted(os.listdir(path))
        finfo = [(f,os.path.getsize(os.path.join(path,f)))
                 for f in files if os.path.isfile(os.path.join(path,f))]
    except:
        finfo = []
    result = { 'files': finfo }
    result['jobid'] = id
    return format_result(result, format=format, template="index.html")

@app.route('/jobs/<int:id>/data/', methods=['GET','POST'])
def putfile(id):
    if flask.request.method=='POST':
        print "warning: XSS attacks possible if stored file is mimetype html"
        file = flask.request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(store.path(id), filename))
    return redirect(url_for('getfile',id=id,filename='index.html'))

@app.route('/jobs/<int:id>/data/<filename>')
def getfile(id, filename):
    as_attachment = filename.endswith('.htm') or filename.endswith('.html')
    if filename.endswith('.json'):
        mimetype = "application/json"
    else:
        mimetype = None

    return send_from_directory(store.path(id), filename,
                               mimetype=mimetype, as_attachment=as_attachment)

#@app.route('/jobs/<int:id>.<format>', methods=['PUT'])
#def update_job(id, format='.json'):
#    """
#    PUT /job/<id>.<format>
#
#    Updates a job using data from the job submission form.
#    """
#    book = Book(id=id, name=u"I don't know") # Your query
#    book.name = request.form['name'] # Save it
#    flash('Book %s updated!' % book.name)
#    return redirect(url_for('show_job', id=id, format=format))

#@app.route('/jobs/new.html')
#def new_job_form():
#    """
#    GET /jobs/new
#
#    Returns a job submission form.
#    """
#    return render_template('new_job.html')

#@app.route('/jobss/<int:id>/edit.html')
#def edit_job_form(id):
#    """
#    GET /books/<id>/edit
#
#    Form for editing job details
#    """
#    book = Book(id=id, name=u'Something crazy') # Your query
#    return render_template('edit_book.html', book=book)


def init_scheduler(conf):
    if conf == 'slurm':
        from slurm import Scheduler
    elif conf == 'direct':
        from simplequeue import Scheduler
    else:
        raise ValueError("unknown scheduler %s"%conf)
    return Scheduler()

def serve():
    global scheduler
    import os

    os.nice(19)
    scheduler = init_scheduler(app.config['SCHEDULER'])
    app.run()

if __name__ == '__main__': serve()
