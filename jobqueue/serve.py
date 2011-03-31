import cPickle as pickle

import flask
from flask import redirect, url_for, flash

app = flask.Flask(__name__)
app.config['DEBUG'] = True
app.config['SECRET_KEY'] = 'secret'

class JobServer(object):
    def __init__(self):
        self.next = 0
        self.jobs = []
        self.info = {}
    def list_jobs(self):
        return self.jobs
    def schedule(self, request):
        id = self.next
        self.next += 1
        request['id'] = id
        self.info[id] = request
        self.jobs.append(id)
        return id
    def delete(self, id):
        # Delete job from queue
        self.info.pop(id,None)
        try:
            self.jobs.remove(id)
        except ValueError:
            pass


job_server = JobServer()
job_server.schedule(dict(name='first request'))
job_server.schedule(dict(name='second request'))

def format_result(result, format='json', template=None):
    """
    Return result as a particular format.  html
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


@app.route('/jobs.<format>')
def list_jobs(format):
    """GET /jobs.<format>

    Return a list of all active job ids."""
    result = dict(jobs=job_server.list_jobs())
    return format_result(result, format, template='list_jobs.html')

@app.route('/jobs.<format>', methods=['POST'])
def new_job(format):
    """POST /jobs.<format>

    Schedule a new job."""
    print "request",flask.request
    print 'data',flask.request.data
    print 'values',flask.request.values.items()
    id = job_server.schedule(flask.request.json)
    flash('Job %d scheduled' % id)
    result = job_server.info[id]
    return format_result(result, format=format, template='show_job.html')

@app.route('/jobs/<int:id>.<format>', methods=['GET'])
def show_job(format, id):
    """GET /jobs/<id>.<format>

    Get job info by id.
    """
    result = job_server.info[id]
    return format_result(result, format=format, template='show_job.html')

@app.route('/jobs/<int:id>.<format>', methods=['DELETE'])
def delete_job(id, format):
    """DELETE /jobs/<id>.<format>

    Deletes a job, returning the list of remaining jobs as <format>"""
    job_server.delete(id)
    flash('Job %s deleted' % id)
    result = dict(jobs=job_server.list_jobs())
    return format_result(result, format=format, template="list_jobs.html")

#@app.route('/jobs/new')
#def new_book():
#    """GET /jobs/new
#
#    The form for a new job"""
#    return render_template('new_job.html')

#@app.route('/books/<int:id>/edit')
#def edit_book(id):
#    """GET /books/<id>/edit
#
#    Form for editing a book"""
#    book = Book(id=id, name=u'Something crazy') # Your query
#    return render_template('edit_book.html', book=book)

#@app.route('/books/<int:id>', methods=['PUT'])
#def update_book(id):
#    """PUT /books/<id>
#
#    Updates a book"""
#    book = Book(id=id, name=u"I don't know") # Your query
#    book.name = request.form['name'] # Save it
#    flash('Book %s updated!' % book.name)
#    return redirect(url_for('show_book', id=book.id))


if __name__ == '__main__':
    app.run()

