import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import sympy
from scipy.optimize import curve_fit

################# constants ##########################
INTTYPE = ['iteration', 'PID', 'EID', 'ray']
HOMEDIR = '/Users/alexaksentyev/REPOS/NICA-FS/'
ELNAMES = np.insert(np.load('nica_element_names.npy'),0,'INJ')
ELNAMES = np.insert(ELNAMES, 1,'RF')

############### function definintions #####################
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

############## class definitions ####################
class Data:
    def __init__(self, path, filename):
        self._data = load_data(path, filename)

    @property
    def data(self):
        return self._data
    @property
    def co(self):
        return self._data[:,0]
    def __getitem__(self, key):
        return self._data[key]

class TSS(Data):
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
