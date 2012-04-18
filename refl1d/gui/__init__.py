"""
Refl1D GUI support
"""
from utilities import resource_dir

def data_files():
    """
    Return the data files associated with the package.

    The format is a list of (directory, [files...]) pairs which can be
    used directly in the py2exe setup script as::

        setup(...,
              data_files=data_files(),
              ...)
    """
    import os, glob
    def _finddata(*patterns):
        path = resource_dir()
        files = []
        for p in patterns:
            files += glob.glob(os.path.join(path,p))
        return files
    data_files = [('refl1d-data', _finddata('*.png','*.ico','*.jpg'))]
    return data_files

def package_data():
    """
    Return the data files associated with the package.

    The format is a dictionary of {'fully.qualified.module', [files...]}
    used directly in the setup script as::

        setup(...,
              package_data=package_data(),
              ...)
    """
    return { 'refl1d.gui':
            ['resources/*.png','resources/*.ico','resources/*.jpg'] }

