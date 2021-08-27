import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, DAVEC, load_data

DIR  = '../data/TEST/'

def load_tss(path=HOMEDIR+DIR+'MU.dat'):
    d_type = [('EL', int), ('PID', int)] + list(zip(['NU', 'NX','NY','NZ'], [float]*4))
    dat = np.loadtxt(path, dtype=d_type)
    nray = len(np.unique(dat['PID']))
    dat.shape = (-1, nray)
    return dat[:, 1:]

if __name__ == '__main__':
    dat = load_data(DIR, 'TRPRAY.dat')
    fig, ax = plt.subplots(1,2)
    ax[0].plot(dat['X'], dat['A'])
    ax[1].plot(dat['Y'], dat['B'])
