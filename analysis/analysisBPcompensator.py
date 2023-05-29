import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import DAVEC, fit_line
from glob import glob
import re

DIR = '../data/BYPASS_SEX_wRC/RC-vary/'
mrkr_form = lambda n: 'CASE_{:d}'.format(n)
case_sign = '*'

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

##### analysis results ########
def plot_tilt_action(tilts0, nu0, n0, nuRLC): # effect of element tilting (nbar, nu as function of mean tilt)
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

def plot_optimizing_results(tilts0, nu0, n0, nuRLC): # restoring nbar=ny after tilts by both RC and LC
    fig, ax = plt.subplots(2,1,sharex=True)
    popt, perr = fit_line(tilts0, nuRLC['RC'])
    ax[0].plot(tilts0, popt[0]+popt[1]*tilts0, '--r')
    ax[0].plot(tilts0, nuRLC['RC'], '.')
    ax[0].set_ylabel(r'required $\nu_{rc}$')
    ax[1].plot(tilts0, n0['X'], 'r.', label=r'$\bar n_x$')
    ax[1].plot(tilts0, n0['Y'], 'g.', label=r'$\bar n_y$')
    ax[1].plot(tilts0, n0['Z'], 'b.', label=r'$\bar n_z$')
    ax[1].set_ylabel(r'effected $\bar n$'); ax[1].legend()
    ax[1].set_xlabel(r'$\langle\theta_{tilt}\rangle$ compensated [deg]')
    for i in range(2):
        ax[i].grid()

    fig1, ax1 = plt.subplots(1,1)
    ax1.plot(nuRLC['RC'], nuRLC['LC'], '.')
    ax1.set_ylabel(r'required (longgitudinal) corrector $\nu$')
    ax1.set_xlabel(r'used (radial) compensator $\nu$')
    ax1.ticklabel_format(style='sci',scilimits=(0,0),useMathText=True, axis='y')
    ax1.grid()

def plot_RC_action(tilts0, nu0, n0, nuRLC): # vary RC at a fixed tilt distribution
    tilts_all = np.loadtxt('../data/BYPASS_SEX_wRC/EB_GAUSS:ALL.in')/np.pi * 180  # rad -> deg
    mean_tilt = tilts_all.mean()
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[0].set_title(r'$\langle\theta_{tilt}\rangle$ = '+'{:4.2e} [deg]'.format(mean_tilt))
    ax[0].plot(nuRLC['RC'], n0['X'],'-.')
    ax[0].set_ylabel(r'$\bar n_x$')
    ax[1].plot(nuRLC['RC'], n0['Y'],'-.')
    ax[1].set_ylabel(r'$\bar n_y$')
    ax[1].ticklabel_format(style='sci',scilimits=(0,0),axis='y',useMathText=True)
    ax[2].plot(nuRLC['RC'], n0['Z'],'-.')
    ax[2].set_ylabel(r'$\bar n_z$')
    ax[2].ticklabel_format(style='sci',scilimits=(0,0),axis='both',useMathText=True)
    ax[2].set_xlabel(r'$\nu_{rc}$')
    for i in range(3):
        ax[i].grid()

    phi_XY = np.arctan(n0['Y']/n0['X'])/np.pi*180
    fig1, ax1 = plt.subplots(1,1)
    ax1.plot(nuRLC['RC'], phi_XY, '-.')
    ax1.set_xlabel(r'$\nu_{rc}$')
    ax1.set_ylabel(r'$\angle(\bar n_x,\bar n_y)$ [deg]')

if __name__ == '__main__':
    cases = [int(re.sub('_','',e[-2:])) for e in glob(DIR+'ABERRATIONS:'+case_sign)]
    cases.sort()
    ncases = len(cases)
    nbar = {}; nu = {}
    n0 = np.zeros(ncases, dtype=list(zip(['X','Y','Z'],[float]*3))); nu0 = np.zeros(ncases)
    tilts = np.zeros((92,ncases))
    nuRLC = np.zeros(ncases, dtype=[('RC', float), ('LC', float)])
    for i, case in enumerate(cases):
        print(case)
        nbar.update({case: NBAR(DIR, mrkr_form(case))})
        nu.update({case: DAVEC(DIR+'NU:'+mrkr_form(case)+'.da')})
        tmp = [nbar[case].mean[e] for e in range(3)]
        n0[i] = tmp[0],tmp[1], tmp[2]
        nu0[i] = nu[case].const
        try:
            tilts[:,i] = np.loadtxt(DIR+'EB_GAUSS:'+mrkr_form(case)+'.in')/np.pi * 180 # rad -> deg
        except:
            None
        tmp = np.loadtxt(DIR+'/LATTICE-PARAMETERS:'+mrkr_form(case),skiprows=1)[3:]
        nuRLC[i] = tmp[0], tmp[1]                                                      
    tilts0 = tilts.mean(axis=0)

    plot_RC_action(tilts0, nu0, n0, nuRLC)

    
