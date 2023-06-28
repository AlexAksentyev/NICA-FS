import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import DAVEC, NBAR, load_data, guess_freq
from glob import glob
import re
from scipy.optimize import curve_fit

DIR = '../data/BYPASS_SEX_wRC/AXIS/LONG/'

Fcyc = 0.2756933208648683e6 # beam revolution frequency [Hz]
TAU = 1/Fcyc # revolution period [sec]

mrkr_form = lambda n: 'CASE_{:d}'.format(n)
case_sign = '*'

def fit_sine(x,y):
    fun = lambda x, a,f,p: a*np.sin(2*np.pi*f*x + p)
    popt, pcov = curve_fit(fun, x, y)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr
def fit_SPN(spdat, pid=0):
    t = spdat[:,pid]['iteration']*TAU
    fun = lambda x, a,f,p: a*np.sin(2*np.pi*f*x + p)
    par, err = {}, {}
    names = ['A','F','P']
    p0dict = {'S_X':0, 'S_Y':0, 'S_Z':1}
    fig, ax = plt.subplots(3,1,sharex=True)
    for i, tag in enumerate(['S_X','S_Y','S_Z']):
        y = spdat[:,pid][tag]
        # initial guesses
        a0 = (np.max(y) - np.min(y))/2
        f0 = guess_freq(t, y)
        print(a0, f0)
        # fit
        popt, pcov = curve_fit(fun, t, y, p0 = (a0,f0,p0dict[tag]))
        perr = np.sqrt(np.diag(pcov))
        par.update({tag:dict(zip(names, popt))})
        err.update({tag:dict(zip(names, perr))})
        # plot
        ax[i].plot(t, y, 'b-')
        ax[i].plot(t, fun(t, *popt), 'r--', label='f = {:4.2f}'.format(popt[1]))
        ax[i].set_ylabel(tag)
        ax[i].legend()
    ax[2].set_xlabel('t [sec]')
    return par, err
def load_tss(dir):
    cases = [int(re.findall(r'\d+',e)[0]) for e in glob(DIR+'ABERRATIONS:'+case_sign)]
    cases.sort()
    #cases = np.arange(20)
    ncases = len(cases)
    nbar = {}; nu = {}
    n0 = np.zeros(ncases, dtype=list(zip(['X','Y','Z'],[float]*3))); nu0 = np.zeros(ncases)
    tilts = np.zeros((ncases, 92))
    for i, case in enumerate(cases):
        print(case)
        nbar.update({case: NBAR(DIR, mrkr_form(case))})
        nu.update({case: DAVEC(DIR+'NU:'+mrkr_form(case)+'.da')})
        tmp = [nbar[case].mean[e] for e in range(3)]
        n0[i] = tmp[0],tmp[1], tmp[2]
        nu0[i] = nu[case].const
        tilts[i] = np.loadtxt(DIR+'GAUSS:'+mrkr_form(case)+'.in')
    return nu, nbar, nu0, n0, tilts

def load_tr(dir):
    cases = [int(re.findall(r'\d+',e)[0]) for e in glob(DIR+'ABERRATIONS:'+case_sign)]
    cases.sort()
    #cases = np.arange(20)
    ncases = len(cases)
    datdict = {}
    psdatdict = {}
    for i, case in enumerate(cases):
        print(case)
        datdict.update({case: load_data(dir, 'TRPRAY:CASE_{}.dat'.format(case))})
        psdatdict.update({case: load_data(dir, 'TRPSPI:CASE_{}.dat'.format(case))})
    return psdatdict, datdict

def fit_const(x,y):
    const = lambda x, c: 0*x + c
    popt, perr = curve_fit(const, x, y)
    popt = popt[0]; perr = perr[0,0]
    return popt, perr

def compute_nbar0(trdat):
    spin0 = trdat[:,0] # pick reference ray only
    A = spin0[:-1] # S_ini;   normalized, hence 
    B = spin0[1:] # S_fin     no need to normalize
    n = np.zeros(A.shape, dtype=list(zip(['X','Y','Z'], [float]*3)))
    # cross product components
    n['X'] = A['S_Y']*B['S_Z']-A['S_Z']*B['S_Y']
    n['Y'] = A['S_Z']*B['S_X']-A['S_X']*B['S_Z']
    n['Z'] = A['S_X']*B['S_Y']-A['S_Y']*B['S_X']
    # normalizing
    nn = np.sqrt(n['X']**2 + n['Y']**2 + n['Z']**2)
    for tag in ['X','Y','Z']:
        n[tag] /= nn
    # angle computation
    theta = np.arccos(
        np.sum(list(A[tag]*B[tag] for tag in ['S_X','S_Y','S_Z']), axis=0)
        ) # rotation angle
    
    return n, theta

def average_nbar(vect, normalize=False):
    x = np.arange(vect['X'].shape[0])
    avg = {}; err = {}
    for tag in ['X','Y','Z']:
        popt, perr = fit_const(x, vect[tag])
        avg.update({tag:popt})
        err.update({tag:perr})
    if normalize:
        norma = np.sqrt(np.sum(avg[tag]**2 for tag in ['X','Y','Z']))
        for tag in ['X','Y','Z']:
            avg[tag] /= norma
    return avg, err

def nbar_analysis(n0tss, trdat, norm_mean=True):
    n0trk, thtrk = compute_nbar0(trdat)
    n0trk_mean, n0trk_err = average_nbar(n0trk, norm_mean)
    t = trdat[1:,0]['iteration']*TAU
    fig, ax = plt.subplots(3,1, sharex=True)
    for i, tag in enumerate(['X','Y','Z']):
        ax[i].plot(t, n0trk[tag],
                       label='tracking',
                       ls='solid')
        ax[i].plot(t, n0trk_mean[tag]+0*t,
                       label='mean (normed)' if norm_mean else 'mean',
                       ls='dashed')
        ax[i].plot(t, n0tss[tag]+0*t,
                       label='tss',
                       ls='dotted')
        ax[i].legend()
        ax[i].set_ylabel(r'$\bar n_{}$'.format(tag))
    ax[2].set_xlabel('t [sec]')
    return fig, ax

def plot_PS(phdat, pid=0):
    ph=phdat[:,pid]
    t = ph['iteration']*TAU

    fig = plt.figure()
    gs = fig.add_gridspec(2,3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_title('pid = {:d}'.format(pid))
    ax1.plot(ph['X']*1000, ph['A']*1000, '.')
    ax1.set_xlabel('X [mm]'); ax1.set_ylabel('A [mrad]')
    ax2.plot(ph['Y']*1000, ph['B']*1000, '.')
    ax2.set_xlabel('Y [mm]'); ax2.set_ylabel('B [mrad]')
    ax3.plot(ph['T'], ph['D'], '.')
    ax3.set_xlabel('T'); ax3.set_ylabel('D')
    for ax in (ax1, ax2, ax3):
        ax.ticklabel_format(style='sci', scilimits=(0,0),useMathText=True,axis='both')
    ax4.plot(t, ph['X']*1000, label='X')
    ax4.plot(t, ph['Y']*1000, label='Y')
    ax4.set_xlabel('t [sec]'); ax4.set_ylabel('trans. coord')
    ax4.legend()
    return fig

def plot_SPN(spdat, pid=0):
    sp=spdat[:,pid]
    t=sp['iteration']*TAU
    
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[0].set_title('pid = {:d}'.format(pid))
    for i, var in enumerate(['S_X','S_Y','S_Z']):
        ax[i].plot(t, sp[var])
        ax[i].set_ylabel(var)
    ax[2].set_xlabel('t [sec]')
    return fig
    

if __name__ == '__main__':
    nu, nbar, nu0, n0, tilts = load_tss(DIR)
    spdat, phdat = load_tr(DIR)
    nbar, theta = compute_nbar0(spdat[3])
    fig, ax = nbar_analysis(n0[3], spdat[3])
    fig = plot_PS(phdat[3])
