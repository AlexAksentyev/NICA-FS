import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import sympy
from scipy.optimize import curve_fit, least_squares

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

def fit_line(x,y): # this is used for evaluating the derivative
    # resid = lambda p, x,y: p[0] + p[1]*x - y
    # # initial parameter estimates
    # a0 = y[0]; b0 = (y[-1]-y[0])/(x[-1]-x[0])
    # # fitting
    # result = least_squares(resid, [a0, b0], args=(x,y), loss='soft_l1', f_scale=.1)
    # popt = result.x
    # # computing the parameter errors
    # J = result.jac
    # pcov = np.linalg.inv(J.T.dot(J))*result.fun.std()
    # perr = np.sqrt(np.diagonal(pcov))
    ## same with curve_fit
    line = lambda x,a,b: a + b*x
    ii = slice(0, None) if len(x)<100 else slice(10,-10)
    popt, pcov = curve_fit(line, x[ii], y[ii])
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

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

def project_spin_nbar(spdata, tssdata, ftype='CO'): # either CO, or mean
                                        # DON'T USE MEAN, I only generate data for one turn here
   def make_nbar_seq(component, repnum):
       num_el = len(component)
       pick_i = np.unique(spdata['EID'][:,0])%num_el
       x = component[pick_i]
       x0 = x[0]
       x = x[1:] #np.append(x[1:], x[:2])[:-1]
       x = np.tile(x, repnum) # x is 1d; [1,2,3,..., 1, 2, 3, ..., 1, 2, 3,... ]
       return np.insert(x, 0, x0)
   def normalize(nbar):
       norm_nbar = np.sqrt(nbar['X']**2 + nbar['Y']**2 + nbar['Z']**2)
       if np.any(abs(norm_nbar-1))>1e-6:
           print('********** nbar norm is suspect! {}, {}'.format(norm_nbar.min(), norm_nbar.max()))
       return {lbl: nbar[lbl]/norm_nbar for lbl in ['X','Y','Z']}
   s = {lbl:spdata['S_'+lbl] for lbl in ['X','Y','Z']}
   ntrn = np.unique(spdata['iteration'][1:,0])[-1]
   if ftype=='CO':
       n = {lbl:make_nbar_seq(tssdata['N'+lbl][:,0], ntrn) for lbl in ['X','Y','Z']}
   elif ftype=='mean':
       n = {lbl:make_nbar_seq(np.mean(tssdata['N'+lbl], axis=1), ntrn) for lbl in ['X','Y','Z']}
   n = normalize(n)
   prod = {lbl: (s[lbl].T*n[lbl]).T for lbl in ['X','Y','Z']}
   return prod['X']+prod['Y']+prod['Z']

def project_spin_axis(spdata, axis=[0,0,1]):
    s = {lbl:spdata['S_'+lbl] for lbl in ['X','Y','Z']}
    n = dict(zip(['X','Y','Z'], axis))
    prod = {lbl: (s[lbl].T*n[lbl]).T for lbl in ['X','Y','Z']}
    return prod['X']+prod['Y']+prod['Z']

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

class Polarization(Data):
    def __init__(self, iteration, eid, value, spin_proj):
        self._data = np.array(list(zip(iteration, eid, value)),
                                  dtype = [('iteration', int), ('EID', int), ('Value', float)])
        self._spin_proj = spin_proj

    @classmethod
    def on_nbar(cls, spdata, tssdata):
        sp_proj = project_spin_nbar(spdata, tssdata, ftype='CO')
        return cls._initializer(spdata, sp_proj)

    @classmethod
    def on_axis(cls, spdata, axis=[0,0,1]):
        sp_proj = project_spin_axis(spdata, axis)
        return cls._initializer(spdata, sp_proj)

    @classmethod
    def _initializer(cls, spdata, sp_proj):
        it = spdata['iteration'][:,0]
        try:
            eid = spdata['EID'][:,0]
        except:
            eid = np.ones(it.shape)
        nray = sp_proj.shape[1]
        pol = sp_proj.sum(axis=1)/nray
        return cls(it, eid, pol, sp_proj)
    
    @property
    def spin_proj(self):
        return self._spin_proj
    @property
    def co(self):
        return self._data
    def plot(self, eid, xlab='sec', L=503, gamma=1.14):
        beta = np.sqrt(1 - 1/gamma**2)
        v = beta*3e8
        tau = L/v
        jj = self['EID']==eid
        y = self['Value'][jj]
        it = self['iteration'][jj]
        t = it*tau
        par, err = fit_line(t, y)
        fig, ax = plt.subplots(1,1)
        if xlab=='sec':
            x = t
        elif xlab=='turn':
            x = it
            par, err =  (tau*e for e in [par, err])
        else:
            x = t
            xlab = 'sec'
        ax.plot(x,y, '.')
        ax.plot(x, par[0] + x*par[1], '-r',
                    label=r'$slp = {:4.2e} \pm {:4.2e}$ [u/{}]'.format(par[1], err[1], xlab))
        ax.set_ylabel(r'$\sum_i(\vec s_i, \bar n_{})$'.format(eid))
        ax.set_xlabel(xlab)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        ax.legend()
        return fig, ax
