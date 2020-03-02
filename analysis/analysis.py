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

## functions loading transfer maps produced by
## MAD-X PTC module
def load_madx(address):
    def parse_names(names):
        conv = lambda x: [int(e) for e in list(x)]
        names = np.array([e.strip('",C').split("_") for e in names])
        fin = np.array([int(e) for e in names[:,0]])-1
        ini = np.array([conv(e) for e in names[:,1]])
        return np.vstack((fin, ini.T)).T
    maptable = np.loadtxt(address, skiprows=8, usecols=(0,1), dtype=[('name', object), ('coef', float)])

    trans_map = np.zeros((6,6))
    pmat = parse_names(maptable['name'])
    carr = maptable['coef']

    for k, c  in enumerate(carr):
        i = pmat[k][0]
        try:
            j = list(pmat[k][1:]).index(1)
            trans_map[i,j] = c
        except:
            print(k, i, c, '***** constant coefficient')
    return trans_map

## COSY PM procedure
def load_cosy(address):
    tmp = np.genfromtxt(address, skip_footer = 1,
                        delimiter=(1, 14, 14, 14, 14, 14, 7),
                        usecols = range(1,6))
    trans_map = np.zeros((6, 6)) 
    trans_map[:5,:] = tmp.T; trans_map[5,-1] = 1
    return trans_map

def pm(map_): ## outputs maps loaded by the above two functions into console in a human-readable form
    conv = lambda x: 0 if abs(x)<1e-12 else x
    for i in range(6):
        row = list(map(conv, map_[i,:]))
        print('{: 4.6e} {: 4.6e} {: 4.6e} {: 4.6e} {: 4.6e} {: 4.6e}'.format(*row))

def compare(m1, m2): # compares maps loaded by load_cosy and load_madx
    print('x-a')
    print('{: 4.6e}|{: 4.6e}    {: 4.6e}|{: 4.6e}'.format(m1[0,0], m1[0,1], m1[1,0], m1[1,1]))
    print('{: 4.6e}|{: 4.6e}    {: 4.6e}|{: 4.6e}'.format(m2[0,0], m2[0,1], m2[1,0], m2[1,1]))
    print('{: 4.6e}|{: 4.6e}    {: 4.6e}|{: 4.6e}'.format(
        m2[0,0]-m1[0,0], m2[0,1]-m1[0,1],
        m2[1,0]-m1[1,0], m2[1,1]-m1[1,1])
              )
    print('=============')
    print('y-b')
    print('{: 4.6e}|{: 4.6e}    {: 4.6e}|{: 4.6e}'.format(m1[2,2], m1[2,3], m1[3,2], m1[3,3]))
    print('{: 4.6e}|{: 4.6e}    {: 4.6e}|{: 4.6e}'.format(m2[2,2], m2[2,3], m2[3,2], m2[3,3]))
    print('{: 4.6e}|{: 4.6e}    {: 4.6e}|{: 4.6e}'.format(
        m2[2,2]-m1[2,2], m2[2,3]-m1[2,3],
        m2[3,2]-m1[3,2], m2[3,3]-m1[3,3]))
    print('=============')
    print('t-d')
    print('{: 4.6e}|{: 4.6e}    {: 4.6e}|{: 4.6e}'.format(m1[4,4], m1[4,5], m1[5,4], m1[5,5]))
    print('{: 4.6e}|{: 4.6e}    {: 4.6e}|{: 4.6e}'.format(m2[4,4], m2[4,5], m2[5,4], m2[5,5]))
    print('{: 4.6e}|{: 4.6e}    {: 4.6e}|{: 4.6e}'.format(
        m2[4,4]-m1[4,4], m2[4,5]-m1[4,5],
        m2[5,4]-m1[5,4], m2[5,5]-m1[5,4]))
    print('=============')
        
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
