from numpy import pi, inf, nan, sqrt, log, degrees, radians, cos, sin, tan
from numpy import arcsin as asin, ceil, clip

_FWHM_scale = sqrt(log(256))
def FWHM2sigma(s):
    return s/_FWHM_scale
def sigma2FWHM(s):
    return s*_FWHM_scale

def QL2T(Q=None,L=None):
    """
    Compute angle from Q and wavelength.

    T = asin( |Q| L / 4 pi )

    Returns T in degrees.
    """
    return degrees(asin(abs(Q) * L / (4*pi)))

def TL2Q(T=None,L=None):
    """
    Compute Q from angle and wavelength.

    Q = 4 pi sin(T) / L

    Returns Q in inverse Angstroms.
    """
    return 4 * pi * sin(radians(T)) / L

def dTdL2dQ(T=None, dT=None, L=None, dL=None):
    """
    Convert wavelength dispersion and angular divergence to Q resolution.

    *T*,*dT*  (degrees) angle and FWHM angular divergence
    *L*,*dL*  (Angstroms) wavelength and FWHM wavelength dispersion

    Returns 1-sigma dQ
    """

    # Compute dQ from wavelength dispersion (dL) and angular divergence (dT)
    T,dT = radians(T), radians(dT)
    #print T, dT, L, dL
    dQ = (4*pi/L) * sqrt( (sin(T)*dL/L)**2 + (cos(T)*dT)**2 )

    #sqrt((dL/L)**2+(radians(dT)/tan(radians(T)))**2)*probe.Q
    return FWHM2sigma(dQ)

def dQdT2dLoL(Q, dQ, T, dT):
    """
    Convert a calculated Q resolution and wavelength divergence to a
    wavelength dispersion.

    *Q*, *dQ* (inv Angstroms)  Q and 1-sigma Q resolution
    *T*, *dT* (degrees) angle and FWHM angular divergence

    Returns FWHM dL/L
    """
    return sqrt( (sigma2FWHM(dQ)/Q)**2 - (radians(dT)/tan(radians(T)))**2 )

def parse_file(file):
    """
    Parse a file into a header and data.

    Header lines look like # key value
    Keys can be made multiline by repeating the key
    Data lines look like float float float
    Comment lines look like # float float float
    Data may contain inf or nan values.
    """
    import numpy
    if hasattr(file, 'readline'):
        fh = file
    elif not string_like(file):
        raise ValueError('file must be a name or a file handle')
    elif file.endswith('.gz'):
        import gzip
        fh = gzip.open(file)
    else:
        fh = open(file)
    header = {}
    data = []
    for line in fh:
        columns,key,value = _parse_line(line)
        if columns:
            data.append([indfloat(v) for v in columns])
        if key:
            if key in header:
                header[key] = "\n".join((header[key],value))
            else:
                header[key] = value
    if fh is not file: fh.close()
    #print [len(d) for d in data]
    #print "\n".join(k+":"+v for k,v in header.items())
    return header, numpy.array(data).T

def string_like(s):
    try: s+''
    except: return False
    return True

def _parse_line(line):
    # Check if line contains comment character
    idx = line.find('#')
    if idx < 0: return line.split(),None,''

    # If comment is after data, ignore the comment
    if idx > 0: return line[:idx].split(),None,''

    # Check if we have '# key value'
    line = line[1:].strip()
    idx = line.find(' ') # should also check for : and =
    if idx < 0: return [],None,None

    # Separate key and value
    key = line[:idx]
    value = line[idx+1:].lstrip()

    # If key is a number, it is simply a commented out data point
    if key[0] in '.-+0123456789': return [], None, None

    # Strip matching quotes off the value
    if (value[0] in ("'",'"')) and value[-1] is value[0]:
        value = value[1:-1]

    return [],key,value

def indfloat(s):
    try:
        return float(s)
    except:
        s = s.lower()
        if s == 'inf': return inf
        if s == '-inf': return -inf
        if s == 'nan': return nan
        raise


# Color functions
def dhsv(color, dh=0, ds=0, dv=0, da=0):
    """
    Modify color on hsv scale.
    
    *dv* change intensity, e.g., +0.1 to brighten, -0.1 to darken.
    *dh* change hue, e.g., 
    """
    from matplotlib.colors import colorConverter
    from colorsys import rgb_to_hsv, hsv_to_rgb
    from numpy import clip, array
    r,g,b,a = colorConverter.to_rgba(color)
    a = rgba[3] if len(rgba) > 3 else 1
    h,s,v = rgb_to_hsv(r,g,b)
    h,s,v,a = [clip(val,0,1) for val in h+dh,s+ds,v+dv,a+da]
    r,b,g = hsv_to_rgb(h,s,v)
    return numpy.array((r,g,b,a))
