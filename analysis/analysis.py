import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import sympy
from scipy.optimize import curve_fit

## constants
PSVARS = ['X','A','Y','B','T','D']
SPVARS = ['S_X','S_Y','S_Z']
PSDTYPE = list(zip(PSVARS, [float]*6))
SPDTYPE = list(zip(SPVARS, [float]*3))

HOMEDIR = '/Users/alexaksentyev/REPOS/NICA-FS/'

ELNAMES = np.insert(np.load('nica_element_names.npy'),0,'INJ')
ELNAMES = np.insert(ELNAMES, 1,'RF')

## function definintions
def _read_header(fileaddress):
    with open(fileaddress) as f:
        nray_line = f.readline()
        dtype_line = f.readline()
    nray = int(nray_line.strip().split(":")[1])
    dtype = dtype_line.strip().split()[1:]
    # i only want the part about turns, elements and particle ids
    dtype = [e for e in dtype if ((e not in PSVARS) and (e not in SPVARS))]
    dtype = list(zip(dtype, [int]*len(dtype)))
    return nray, dtype

def _shape_up(dat, nrays):
    dat.shape = (-1, nrays)
    dat = dat[:, 1:]
    return dat
    
def load_ps(path, filename='TRPRAY.dat', ndim=3):
    nray, d_type = _read_header(path+filename)
    d_type += PSDTYPE[:2*ndim]
    ps = np.loadtxt(path+filename, d_type, skiprows=2)
    ps = _shape_up(ps, nray)
    return ps

def load_sp(path, filename='TRPSPI.dat'):
    nray, d_type = _read_header(path+filename)
    d_type += SPDTYPE
    sp = np.loadtxt(path+filename, d_type, skiprows=2)
    sp = _shape_up(sp, nray)
    return sp

def tick_labels(dat, name=True):
    if dat.ndim>1:
        it = dat['iteration'][:,0]
        eid = dat['EID'][:,0]
    else:
        it = dat['iteration']
        eid = dat['EID']
    nit = np.unique(it[1:])
    elname = ELNAMES[eid]
    if name:
        res = ['{} ({}:{})'.format(*e) for e in list(zip(elname, it, eid))]
    else:
        res = ['({}:{})'.format(*e) for e in list(zip(it, eid))]
    return res
        
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

class Particle:
    def __init__(self, path, name):
        self._name = name
        self._ps = load_ps(path, 'TRPRAY.dat')
        self._sp = load_sp(path, 'TRPSPI.dat')

    def plot_spin(self, pcl_ids=slice(0,None), elem_ids=slice(0,None), savedir='img', name=''):
        dat = self._sp[:,pcl_ids]
        fig, ax = plt.subplots(3,1,sharex=True)
        ax[0].plot(dat['S_X']); ax[0].set_ylabel('S_X')
        ax[0].grid(axis='x')
        ax[1].plot(dat['S_Y']); ax[1].set_ylabel('S_Y')
        ax[1].grid(axis='x')
        ax[2].plot(dat['S_Z']); ax[2].set_ylabel('S_Z')
        ax[2].grid(axis='x')
        ax[2].set_xlabel('(TURN, EID)')
        ax[0].set_title(self._name.upper())
        lbls = tick_labels(dat, False)
        tks = np.arange(dat.shape[0])
        plt.xticks(ticks=tks[elem_ids], labels=lbls[elem_ids], rotation=90)
        if name!='':
            name = '-'+name
        plt.savefig("{}/spin-plot-{}{}.png".format(savedir,self._name, name),
                        bbox_inches = 'tight', pad_inches = 0.1,
                        dpi=600)
    def plot_ps(self, varx, vary, turns):
        fig2, ax2 = plt.subplots(1,1)
        ax2.plot(self._ps[varx][turns], self._ps[vary][turns], '.')
        ax2.set_ylabel(vary)
        ax2.set_xlabel(varx)
        ax2.ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)

    @property
    def ps(self):
        return self._ps
    @property
    def sp(self):
        return self._sp
