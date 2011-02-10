from __future__ import division
import os
import sys
import shutil
import wx

def load_problem(args):
    file, options = args[0], args[1:]
    ctx = dict(__file__=file)
    argv = sys.argv
    sys.argv = [file] + options
    execfile(file, ctx) # 2.x
    sys.argv = argv
    try:
        problem = ctx["problem"]
    except AttributeError:
        raise ValueError(file+" does not define 'problem=FitProblem(models)'")
    problem.file = file
    problem.options = options
    return problem


def make_store(problem, opts):
        # Determine if command line override
        if opts.store != None:
            problem.store = opts.store
        problem.output = os.path.join(problem.store,problem.name)

        # Check if already exists
        if not opts.overwrite and os.path.exists(problem.output+'.out'):
            if opts.batch:
                print >>sys.stderr, problem.output+" already exists.  Use -overwrite to replace."
                sys.exit(1)
            msg_dlg = wx.MessageDialog(None,str(problem.store)+" Already exists. Press 'yes' to overwrite, or 'No' to abort and restart with newpath",'Overwrite Directory',wx.YES_NO | wx.ICON_QUESTION)
            retCode = msg_dlg.ShowModal()
            if (retCode != wx.ID_YES):
                sys.exit(1)
            msg_dlg.Destroy()

        # Create it and copy model
        try: os.mkdir(problem.store)
        except: pass
        shutil.copy2(problem.file, problem.store)
