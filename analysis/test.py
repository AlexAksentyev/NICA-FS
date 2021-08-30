import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, DAVEC, load_data

DIR  = '../data/TEST/FIRST-ST/'

def load_tss(path=HOMEDIR+DIR+'MU.dat'):
    d_type = [('EL', int), ('PID', int)] + list(zip(['NU', 'NX','NY','NZ'], [float]*4))
    dat = np.loadtxt(path, dtype=d_type)
    nray = len(np.unique(dat['PID']))
    dat.shape = (-1, nray)
    return dat[:, 1:]

if __name__ == '__main__':
    dat = load_data(DIR, 'TRPRAY.dat')
    spdat = load_data(DIR, 'TRPSPI.dat')
    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(dat[:,1]['X'], dat[:,1]['A'])
    ax[0,1].plot(dat[:,1]['Y'], dat[:,1]['B'])
    ax[1,0].plot(dat[:,3]['X'], dat[:,3]['A'])
    ax[1,1].plot(dat[:,3]['Y'], dat[:,3]['B'])
