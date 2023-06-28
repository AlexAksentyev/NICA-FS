import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import DAVEC, NBAR
from glob import glob
import re

DIR = '../data/BYPASS_SEX_wRC/RESFLIP-GAMMA/'
par_name = 'DGAMMA'
pardict = {'EBE':2, 'DGAMMA':3}
xlabdict = {'RCNU':r'$\nu_{RC}$',
            'EBE':'EB E-field [kV/cm]',
            'DGAMMA':r'beam $\Delta\gamma$'} # in TSS vs PAR
mrkr_form = lambda n: 'CASE_{:d}'.format(n)
case_sign = '*'


def load_data(dir):
    cases = [int(re.findall(r'\d+',e)[0]) for e in glob(DIR+'ABERRATIONS:'+case_sign)]
    cases.sort()
    #cases = np.arange(10)
    ncases = len(cases)
    nbar = {}; nu = {}
    n0 = np.zeros(ncases, dtype=list(zip(['X','Y','Z'],[float]*3))); nu0 = np.zeros(ncases)
    par = np.zeros(ncases)
    for i, case in enumerate(cases):
        print(case)
        nbar.update({case: NBAR(DIR, mrkr_form(case))})
        nu.update({case: DAVEC(DIR+'NU:'+mrkr_form(case)+'.da')})
        tmp = [nbar[case].mean[e] for e in range(3)]
        n0[i] = tmp[0],tmp[1], tmp[2]
        nu0[i] = nu[case].const
        par[i] = np.loadtxt(DIR+'/LATTICE-PARAMETERS:'+mrkr_form(case),skiprows=1)[pardict[par_name]]
    return nu, nbar, nu0, n0, par

def plot_nu_vs_EB(EBe, nu0):
    fig, ax = plt.subplots(1,1)
    ax.plot(EBe, nu0)
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both',useMathText=True)
    ax.set_xlabel('EB E-field [kV/cm]')
    ax.set_ylabel(r'$\nu_0$ spin tune')
    return fig, ax
def plot_nbar_vs_EB(EBe, n0):
    fig, ax = plt.subplots(3,1,sharex=True)
    for i, crd in enumerate(['X','Y','Z']):
        ax[i].plot(EBe, n0[crd])
        ax[i].set_ylabel(r'$\bar n_{}$'.format(crd))
        ax[i].ticklabel_format(style='sci', scilimits=(0,0), axis='both',useMathText=True)
    ax[2].set_xlabel('EB E-field [kV/cm]')
    return fig, ax

def plot_TSS_vs_PAR(par, nu0, n0):
    fig, ax = plt.subplots(4,1,sharex=True)
    color = ['k','g','r','b']
    ax[0].plot(par, nu0, color=color[0])
    ax[0].set_ylabel(r'$\nu_0$')
    ax[0].ticklabel_format(style='sci', scilimits=(0,0), axis='both',useMathText=True)
    ax[0].grid()
    for i, crd in enumerate(['X','Y','Z']):
        ax[i+1].plot(par, n0[crd], color=color[i+1])
        ax[i+1].set_ylabel(r'$\bar n_{}$'.format(crd))
        ax[i+1].ticklabel_format(style='sci', scilimits=(0,0), axis='both',useMathText=True)
        ax[i+1].grid()
    ax[3].set_xlabel(xlabdict[par_name])
    return fig, ax

if __name__ == '__main__':
    nu, nbar, nu0, n0, par = load_data(DIR)
    tilts = np.loadtxt('../data/BYPASS_SEX_wRC/EB_GAUSS:RESFLIP.in')/np.pi * 180  # rad -> deg
    mean_tilt = tilts.mean()
    tilt_sd = tilts.std()
    fig, ax = plot_TSS_vs_PAR(par, nu0, n0)
    ax[0].set_title(r'tilts: {:4.2e} $\pm$ {:4.2e} [deg]'.format(mean_tilt, tilt_sd))
