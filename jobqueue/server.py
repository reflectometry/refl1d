import os, sys
import cPickle as pickle
import flask
from flask import redirect, url_for, flash
from flask import send_from_directory
from werkzeug import secure_filename

from jobqueue import store

app = flask.Flask(__name__)

app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'secret'
#app.config['SCHEDULER'] = 'slurm'
#app.config['SCHEDULER'] = 'direct'
app.config['SCHEDULER'] = 'dispatch'

scheduler = None

def _format_response(response, format='json', template=None):
    """
    Return response as a particular format.
    """
    #print "response",response
    if format == 'html':
        if template is None: flask.abort(400)
        return flask.render_template(template, **response)
    elif format == 'json':
        return flask.jsonify(**dict((str(k),v) for k,v in response.items()))
    elif format == 'pickle':
        return pickle.dumps(response)
    else:
        flask.abort(400) # Bad request


@app.route('/jobs.<format>', methods=['GET'])
def list_jobs(format='json'):
    """
    GET /jobs.<format>

    Return a list of all job ids.
    """
    response = dict(jobs=scheduler.jobs())
    return format_response(response, format, template='list_jobs.html')

@app.route('/jobs/<any(u"pending",u"active",u"error",u"complete"):status>.<format>',
           methods=['GET'])
def filter_jobs(status, format='json'):
    """
    GET /jobs/<pending|active|error|complete>.<format>

    Return all jobs with a particular status.
    """
    response = dict(jobs=scheduler.jobs(status=str(status).upper()))
    return _format_response(response, format, template='list_jobs.html')

@app.route('/jobs.<format>', methods=['POST'])
def create_job(format='json'):
    """
    POST /jobs.<format>

    Schedule a new job, return the job record.

    The POST data should contain::

        {
        notify: "<user@email or @twitterid>",
        service: "<name of service>",
        version: "<service version>",
        name: "<request name>",
        data: "<service data>",
        ...
        }

    The response contains::

        {
        id: <job id>,
        job: <job details>
        }

    Job details is simply a copy of the original request.

    """
    request = flask.request.json
    if request is None: flask.abort(415) # Unsupported media
    id = scheduler.submit(request, origin=flask.request.remote_addr)
    flash('Job %s scheduled' % id)
    response = {'id': id, 'job': scheduler.info(id)}
    #return redirect(url_for('show_job', id=id, format=format))
    return _format_response(response, format=format, template='show_job.html')

@app.route('/jobs/<int:id>.<format>', methods=['GET'])
def show_job(id, format='json'):
    """
    GET /jobs/<id>.<format>

    Get job record by id.

    The response contains::

        {
        id: <job id>,
        job: <job details>
        }

    Job details is simply a copy of the original request.
    """
    response = {'id': id, 'job': scheduler.info(id)}
    return _format_response(response, format=format, template='show_job.html')

@app.route('/jobs/<int:id>/results.<format>', methods=['GET'])
def get_results(id, format='json'):
    """
    GET /jobs/<id>/results.<format>

    Get job results by id.

    Returns::

        {
        id: <job id>
        status: 'PENDING|ACTIVE|COMPLETE|ERROR|UNKNOWN',
        result: <job value>     (absent if status != COMPLETE)
        trace: <error trace>    (absent if status != ERROR)
        }
    """
    response = scheduler.results(id)
    response['id'] = id
    #print "returning response",response
    return _format_response(response, format=format)

@app.route('/jobs/<int:id>/status.<format>', methods=['GET'])
def get_status(id, format='json'):
    """
    GET /jobs/<id>/status.<format>

    Get job status by id.

    Returns::

        {
        id: <job id>,
        status: 'PENDING|ACTIVE|COMPLETE|ERROR|UNKNOWN'
        }
    """
    response = { 'status': scheduler.status(id) }
    response['id'] = id
    return _format_response(response, format=format)


@app.route('/jobs/<int:id>.<format>', methods=['DELETE'])
def delete_job(id, format='json'):
    """
    DELETE /jobs/<id>.<format>

    Deletes a job, returning the list of remaining jobs as <format>
    """
    scheduler.delete(id)
    flash('Job %s deleted' % id)
    response = dict(jobs=scheduler.jobs())
    return _format_response(response, format=format, template="list_jobs.html")
    #return redirect(url_for('list_jobs', id=id, format=format))

@app.route('/jobs/nextjob.<format>', methods=['POST'])
def fetch_work(format='json'):
    import json
    # TODO: verify signature
    request = flask.request.json
    if request is None: flask.abort(415) # Unsupported media
    job = scheduler.nextjob(queue=request['queue'])
    return _format_response(job, format=format)

@app.route('/jobs/<int:id>/postjob', methods=['POST'])
def return_work(id):
    # TODO: verify signature
    request = flask.request.json
    if request is None: flask.abort(415) # Unsupported media
    scheduler.postjob(id, request)
    # Should be signalling code 204: No content
    return _format_response({},format="json")

@app.route('/jobs/<int:id>/data/index.<format>')
def listfiles(id, format):
    try:
        path = store.path(id)
        files = sorted(os.listdir(path))
        finfo = [(f,os.path.getsize(os.path.join(path,f)))
                 for f in files if os.path.isfile(os.path.join(path,f))]
    except:
        finfo = []
    response = { 'files': finfo }
    response['id'] = id
    return _format_response(response, format=format, template="index.html")

@app.route('/jobs/<int:id>/data/', methods=['GET','POST'])
def putfile(id):
    if flask.request.method=='POST':
        # TODO: verify signature
        print "warning: XSS attacks possible if stored file is mimetype html"
        for file in flask.request.files.getlist('file'):
            if not file: continue
            filename = secure_filename(os.path.split(file.filename)[1])
            print "saving",filename
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
        print "Warning: direct scheduler is not a good choice!"
        os.nice(19)
        from simplequeue import Scheduler
    elif conf == 'dispatch':
        from dispatcher import Scheduler
    else:
        raise ValueError("unknown scheduler %s"%conf)
    return Scheduler()

def serve():
    global scheduler
    import os

    scheduler = init_scheduler(app.config['SCHEDULER'])
    app.run()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        app.config['SCHEDULER'] = sys.argv[1]
    serve()
