import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, DAVEC, load_ps, load_sp

DIR  = 'data/TEST/SPINTUNE/'

def load_tss(path=HOMEDIR+DIR+'MU.dat'):
    d_type = [('EL', int), ('PID', int)] + list(zip(['NU', 'NX','NY','NZ'], [float]*4))
    dat = np.loadtxt(path, dtype=d_type)
    nray = len(np.unique(dat['PID']))
    dat.shape = (-1, nray)
    return dat[:, 1:]

if __name__ == '__main__':
    dat = load_tss()
