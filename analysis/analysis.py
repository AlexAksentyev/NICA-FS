import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import sympy
from scipy.optimize import curve_fit

## constants
INTTYPE = ['iteration', 'PID', 'EID', 'ray']
HOMEDIR = '/Users/alexaksentyev/REPOS/NICA-FS/'
ELNAMES = np.insert(np.load('nica_element_names.npy'),0,'INJ')
ELNAMES = np.insert(ELNAMES, 1,'RF')

## function definintions
def _read_header(fileaddress):
    with open(fileaddress) as f:
        nray_line = f.readline()
        dtype_line = f.readline()
    number = nray_line.strip().split(":")[1]
    nray = int(number) if number.isdigit() else int(number.split()[0])
    dtype = dtype_line.strip().split()[1:]
    for i, e in enumerate(dtype):
        if (e in INTTYPE):
            dtype[i] = (e, int)
        else:
            dtype[i] = (e, float)
    return nray, dtype

def _shape_up(dat, nrays):
    dat.shape = (-1, nrays)
    dat = dat[:, 1:]
    return dat
    
def load_data(path, filename):
    nray, d_type = _read_header(path+filename)
    ps = np.loadtxt(path+filename, d_type, skiprows=2)
    ps = _shape_up(ps, nray)
    return ps

class TSS_data:
    def __init__(self, path, filename='MU.dat'):
        self._data = load_data(path, filename)

    @property
    def data(self):
        return self._data
    @property
    def co(self):
        return self._data[:,0]

    def __getitem__(self, key):
        return self._data[key]

    def plot(self, fun=lambda x: x[:,0]):
        norma = np.sqrt(self['NY']**2 + self['NZ']**2)
        sin_psi = self['NY']/norma
        psi = np.rad2deg(np.arcsin(sin_psi))
        fig, ax = plt.subplots(3,1, sharex=True)
        ax[0].plot(fun(self['NU']))
        ax[0].set_ylabel(r'$f(\nu_s)$')
        ax[1].set_ylabel(r'$f(\bar n_{\alpha})$')
        for v in ['NX','NY','NZ']:
            ax[1].plot(fun(self[v]), label=v)
        ax[1].legend()
        ax[2].plot(fun(psi))
        ax[2].set_ylabel(r'$\angle(\bar n,\vec v)$ [deg]')
        for i in range(3):
            ax[i].ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)
            ax[i].grid(axis='x')
        return fig, ax

# def load_sp(path, filename='TRPSPI.dat'):
#     nray, d_type = _read_header(path+filename)
#     sp = np.loadtxt(path+filename, d_type, skiprows=2)
#     sp = _shape_up(sp, nray)
#     return sp

# def load_tss(path, filename):
#     nray, 
#     d_type = [('EID', int), ('PID', int)] + list(zip(['NU', 'NX','NY','NZ'], [float]*4))
#     dat = np.loadtxt(path, dtype=d_type)
#     nray = len(np.unique(dat['PID']))
#     dat.shape = (-1, nray)
#     return dat[:, 1:], case

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

# class Particle:
#     def __init__(self, path, name):
#         self._name = name
#         self._ps0= load_ps()
#         self._ps = load_ps(path, 'TRPRAY.dat')
#         self._sp = load_sp(path, 'TRPSPI.dat')

#     def plot_spin(self, pcl_ids=slice(0,None), elem_ids=slice(0,None), savedir='img', name=''):
#         dat = self._sp[:,pcl_ids]
#         fig, ax = plt.subplots(3,1,sharex=True)
#         ax[0].plot(dat['S_X']); ax[0].set_ylabel('S_X')
#         ax[0].grid(axis='x')
#         ax[1].plot(dat['S_Y']); ax[1].set_ylabel('S_Y')
#         ax[1].grid(axis='x')
#         ax[2].plot(dat['S_Z']); ax[2].set_ylabel('S_Z')
#         ax[2].grid(axis='x')
#         ax[2].set_xlabel('(TURN, EID)')
#         lbls = tick_labels(dat, False)
#         tks = np.arange(dat.shape[0])
#         plt.xticks(ticks=tks[elem_ids], labels=lbls[elem_ids], rotation=90)
#         if name!='':
#             name = '-'+name
#         plt.savefig("{}/spin-plot-{}{}.png".format(savedir,self._name, name),
#                         bbox_inches = 'tight', pad_inches = 0.1,
#                         dpi=600)
#         return fig, ax
    
#     def plot_ps(self, varx, vary, turns, pcl_ids):
#         fig2, ax2 = plt.subplots(1,1)
#         ax2.plot(self._ps[varx][turns, pcl_ids], self._ps[vary][turns, pcl_ids], '.')
#         ax2.set_ylabel(vary)
#         ax2.set_xlabel(varx)
#         ax2.ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)

#     @property
#     def ps(self):
#         return self._ps
#     @property
#     def sp(self):
#         return self._sp


    
def navigators(nu, psi, detector='MPD', gamma=1.14, G=-.142987):
    ''' yields Kz1, Kz2 navigator solenoid strengths
    given the required spin tune and polarization angle in MPD (SPD)'''
    Lz1, Lz2 = .7, .4
    alpha = 0.039984488639998664
    phix = gamma*G*alpha
    psi = np.deg2rad(psi)
    psi = psi if detector=='MPD' else gamma*G*np.pi-psi
    phiz1 = np.pi * nu * np.cos(psi)
    phiz2 = np.pi * nu * np.sin(psi)/np.sin(phix)
    return [e/(1+G) for e in [phiz1/Lz1, phiz2/Lz2]]
