import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import sympy

## constants
PSVARS = ['X','A','Y','B','T','D']
SPVARS = ['S_X','S_Y','S_Z']
PSDTYPE = list(zip(PSVARS, [float]*6))
SPDTYPE = list(zip(SPVARS, [float]*3))

HOMEDIR = '/Users/alexaksentyev/REPOS/NICA-FS/'

## function definintions
def _read_header(fileaddress):
    with open(fileaddress) as f:
        nray_line = f.readline()
        dtype_line = f.readline()
    nray = int(nray_line.strip().split(":")[1])
    dtype = dtype_line.strip().split()[1:]
    dtype = [e for e in dtype if e not in PSVARS]
    dtype = list(zip(dtype, [float]*len(dtype)))
    return nray, dtype

def _shape_up(dat, nrays):
    dat.shape = (-1, nrays)
    dat = dat[:, 1:]
    return dat
    
def load_ps(path, filename='TRPRAY.dat'):
    nray, d_type = _read_header(path+filename)
    ps = np.loadtxt(path+filename, d_type, skiprows=2)
    ps = _shape_up(ps, nray)
    return ps

def load_sp(path, filename='TRPSPI.dat'):
    nray, d_type = _read_header(path+filename)
    sp = np.loadtxt(path+filename, d_type, skiprows=2)
    sp = _shape_up(sp, nray)
    return sp

## class definitions
class DAVEC:
    VARS = ['X','A','Y','B','T','D']
    def __init__(self, path):
        X,A,Y,B,T,D = sympy.symbols(self.VARS)
        self._dtype = list(zip(['i', 'coef', 'ord'] + self.VARS, [int]*9))
        self._dtype[1] = ('coef', float)
        self._data = np.loadtxt(path, skiprows=1,  dtype=self._dtype, comments='-----')
        self.const = self._data[0]['coef']
        cc = self._data['coef']
        e = {}
        for var in self.VARS:
            e[var] = self._data[var]
        expr = cc*(X**e['X'] * A**e['A'] * Y**e['Y'] * B**e['B'] * T**e['T'] * D**e['D'])
        self.coefs = cc
        self.expr = expr
        self.poly = sympy.poly(expr.sum()) # wanted to improve this with list and Poly, but
        # "list representation is not supported," what?
        
    def __call__(self, ps_arr):
        # vals = np.array([self.poly(*v) for v in ps_arr]) # for some reason this doesn't work properly
        vals = np.array([self.poly.eval(dict(zip(ps_arr.dtype.names, v))) for v in ps_arr]) # this does
        return vals

    def __sub__(self, other):
        return self.poly.sub(other.poly)

    def __add__(self, other):
        return self.poly.add(other.poly)
