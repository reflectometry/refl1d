.. _distributed-service:

**********************************
Distributed computing architecture
**********************************

Refl-1D jobs can be submitted to a remote batch queue for processing.  This
allows users to share large clusters for faster processing of the data.  The
queue consists of several components.

* job controller

   http service layer which allows users to submit jobs and view results

* queue

   cluster management layer which distributes jobs to the working nodes

* worker

   process monitor which runs a job on the working nodes

* mapper

   mechanism for evaluating R(x_i) for different x_i on separate CPUs

If you are setting up a local cluster for performing reflectometry
fits, then you will need to read this section, otherwise you can
continue to the next section.

***************************
Job controller installation
***************************

:mod:`jobqueue` is an independent package within refl1d.  It implements
an http API for interacting with jobs.

It is implemented as a WSGI python application using
`Flask <http://flask.pocoo.org>`_

********************
Cluster installation
********************


One approach to securely providing public facing web services is to use
AppArmor to limit the capabilities of the worker process.  Documentation
for `AppArmor on Ubuntu <https://help.ubuntu.com/community/AppArmor>`_ and
`AppArmor on SUSE <http://doc.opensuse.org/products/opensuse/openSUSE/opensuse-security/cha.apparmor.profiles.html>`_
is useful.

Delete a profile::

   sudo rm /etc/apparmor.d/path.to.application
   sudo service apparmor restart

Reload a profile::

   sudo apparmor_parser -r /etc/apparmor.d/path.to.application

Start tracing for a profile with::

   sudo genprof /path/to/application

Then run::

   /path/to/app

Then return to tracing genprof and hit 'S' to scan /var/log/syslog, etc.,
and generate a profile based on the resources it tries to access.

Assuming that the refl1d server is installed as user reflectometry in
a virtualenv of ~/reflserv, MPLCONFIGDIR is set appropriately in the
environment and the reflworkerd workspace has been set, the example
profile given below provides much of the required protection::

    #include <tunables/global>

    /home/reflectometry/reflenv/bin/reflworkerd {
     #include <abstractions/base>
     #include <abstractions/python>

     /bin/dash cx,
     /home/reflectometry/reflserv/bin/python cx,
     /home/reflectometry/reflserv/** r,
     /home/reflectometry/reflserv/**.{so,pyd} mr,
     /home/reflectometry/reflserv/.matplotlib/* rw,
     /home/reflectometry/reflserv/tmp/** rw,
    }

The rw access to tmp and .matplotlib is potentially problematic.  Hostile
models can interfere with each other if they are running at the same time.
Ideally these would be restricted to tmp/jobid/** but this author does not
know how to do so while running with minimal privileges.
