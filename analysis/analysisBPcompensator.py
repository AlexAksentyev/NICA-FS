import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import DAVEC

DIR = '../data/BYPASS_SEX_wRC/laxer-QFS/'
MRKR = 'RADC_BASE'

class NBAR:
    def __init__(self, folder, mrkr):
        self._dict = {}
        for i, lbl in [(1,'X'),(2,'Y'),(3,'Z')]:
            self._dict.update({lbl:DAVEC(folder+'NBAR{:d}:{}.da'.format(i, mrkr))})
        self._mean = np.array([self._dict[e].const for e in ['X','Y','Z']])
        self._norm = np.sqrt(np.sum(self._mean**2))
    @property
    def mean(self):
        return self._mean
    @property
    def norm(self):
        return self._norm


if __name__ == '__main__':
    cases = list(range(25))
    ncases = len(cases)
    nbar = {}; nu = {}
    n0 = np.zeros(ncases, dtype=list(zip(['X','Y','Z'],[float]*3))); nu0 = np.zeros(ncases)
    tilts = np.zeros((12,ncases))
    for i, case in enumerate(cases):
        nbar.update({case: NBAR(DIR, 'MEAN_{:d}'.format(case))})
        nu.update({case: DAVEC(DIR+'NU:MEAN_{:d}.da'.format(case))})
        tmp = [nbar[case].mean[e] for e in range(3)]
        n0[i] = tmp[0],tmp[1], tmp[2]
        nu0[i] = nu[case].const
        tilts[:,i] = np.loadtxt(DIR+'EB_GAUSS:MEAN_{:d}.in'.format(i))/np.pi * 180 # rad -> deg
    tilts0 = tilts.mean(axis=0)
    
    fig, ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(tilts0, nu0, '-.')
    ax[0].set_ylabel(r'$\nu_0$')
    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0),useMathText=True)
    ax[1].plot(tilts0, n0['X'], '-.')
    ax[1].set_xlabel(r'$\langle\theta_{tilt}\rangle$ [deg]')
    ax[1].set_ylabel(r'$\bar n_x$')

    fig, ax1 = plt.subplots(2,1)
    ax1[0].plot(n0['X'], n0['Y'], '-.')
    ax1[0].set_xlabel(r'$\bar n_x$')
    ax1[0].set_ylabel(r'$\bar n_y$')
    phi_XY = np.arctan(n0['Y']/n0['X'])/np.pi*180
    ax1[1].plot(tilts0, phi_XY, '-.')
    ax1[1].set_xlabel(r'$\langle\theta_{tilt}\rangle$ [deg]')
    ax1[1].set_ylabel(r'$\angle(\bar n_x,\bar n_y)$')
