import cPickle as pickle
import flask
from flask import redirect, url_for, flash

app = flask.Flask(__name__)
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'secret'

def format_result(result, format='json', template=None):
    """
    Return result as a particular format.
    """
    if format == 'html':
        if template is None: flask.abort(400)
        return flask.render_template(template, **result)
    elif format == 'json':
        return flask.jsonify(**result)
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
    result = dict(jobs=job_server.list_jobs())
    return format_result(result, format, template='list_jobs.html')

@app.route('/jobs/<any(u"pending",u"active",u"error",u"complete"):status>.<format>',
           methods=['GET'])
def filter_jobs(status, format='json'):
    """
    GET /jobs/<pending|active|error|complete>.<format>

    Return all jobs with a particular status.
    """
    result = dict(jobs=job_server.list_jobs(status=str(status).upper()))
    return format_result(result, format, template='list_jobs.html')


@app.route('/jobs.<format>', methods=['POST'])
def create_job(format='json'):
    """
    POST /jobs.<format>

    Schedule a new job, return the job record.
    """
    id = job_server.submit(flask.request.json)
    flash('Job %s scheduled' % id)
    result = job_server.info(id)
    #return redirect(url_for('show_job', id=id, format=format))
    return format_result(result, format=format, template='show_job.html')

@app.route('/jobs/<int:id>.<format>', methods=['GET'])
def show_job(id, format='json'):
    """
    GET /jobs/<id>.<format>

    Get job record by id.
    """
    result = job_server.info(id)
    return format_result(result, format=format, template='show_job.html')

@app.route('/jobs/<int:id>/result.<format>', methods=['GET'])
def get_results(id, format='json'):
    """
    GET /jobs/<id>/result.<format>

    Get job results by id.

    Returns { status: 'PENDING|ACTIVE|COMPLETE|ERROR|UNKNOWN' }
    If status is 'ERROR', then results['trace'] is the traceback.
    If status is 'COMPLETE', then result will contain the results
    of the job, which vary depending on the service requested.
    """
    result = job_server.results(id)
    print "returning result",result
    return format_result(result, format=format)

@app.route('/jobs/<int:id>/status.<format>', methods=['GET'])
def get_status(id, format='json'):
    """
    GET /jobs/<id>/status.<format>

    Get job status by id.

    Returns { status: 'PENDING|ACTIVE|COMPLETE|ERROR|UNKNOWN' }
    """
    result = job_server.status(id)
    return format_result(result, format=format)


@app.route('/jobs/<int:id>.<format>', methods=['DELETE'])
def delete_job(id, format='json'):
    """
    DELETE /jobs/<id>.<format>

    Deletes a job, returning the list of remaining jobs as <format>
    """
    job_server.delete(id)
    flash('Job %s deleted' % id)
    result = dict(jobs=job_server.list_jobs())
    return format_result(result, format=format, template="list_jobs.html")
    #return redirect(url_for('list_jobs', id=id, format=format))

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


def serve():
    global job_server
    import os
    import sys

    os.nice(10)
    from jobqueue.simplequeue import JobQueue
    job_server = JobQueue()
    import refl1d.fitservice # Registers 'fitter' service

    app.run()

if __name__ == '__main__': serve()
