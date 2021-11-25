import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import load_data, Polarization

LATTICE = 'SECOND-ST'
ENERGY = '130' # MeV

def dotprod(s0, s1):
    return np.sum([s0['S_'+lbl]*s1['S_'+lbl] for lbl in ['X','Y','Z']], axis=0)

def spin_disp(spdat):
    nray = spdat.shape[1]
    s0 = spdat[:,0].repeat(nray-1)
    s0.shape = (-1, nray-1)
    cos_phi = dotprod(s0, spdat[:,1:]) # spdat columns are unit-vectors
    disp = cos_phi.std(axis=1)
    return disp

def load(case, spin_psi='=PSInavi'):
    folder = '../data/'+LATTICE+'/'+ENERGY+'MeV/NAVI-VARI/RATE_'+case+'/'
    dat = load_data(folder, 'TRPRAY:PSI0spin{}.dat'.format(spin_psi))
    spdat = load_data(folder, 'TRPSPI:PSI0spin{}.dat'.format(spin_psi))
    muarr = load_data(folder, 'TRMUARR:PSI0spin{}.dat'.format(spin_psi))
    rate = np.diff(np.rad2deg(np.arcsin(muarr['NY'][:,0])))
    print('psi <rate-of-change> = ', rate.mean(), '[deg/turn]')
    return dat, spdat, muarr

def process(case_name):
    dat, spdat, muarr = load(case_name)
    disp = spin_disp(spdat)
    s0 = spdat[:,0] # reference spin vector
    if True:
        fig1, ax1 = plt.subplots(3,1,sharex=True)
        ax1[0].set_title(r'$\dot \psi = $ {} [deg/turn]'.format(case_name))
        ax1[2].set_xlabel('TURN #')
        for i, var in enumerate(['X', 'Y', 'Z']):
            ax1[i].plot(muarr['TURN'], muarr['N'+var], '--')
            ax1[i].plot(s0['TURN'], s0['S_'+var], '-k')
            ax1[i].set_ylabel(r'$\bar n_{}$ [$\vec s_0$]'.format(var))
    return disp

def main(case_names):
    disp_dict = {}
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('TURN #')
    ax.set_ylabel(r'$\sigma[\cos(\vec s, \vec s_0)]$')
    for i, case in enumerate(case_names):
        disp = process(case)
        disp_dict.update({case:disp})
        ax.plot(disp, label=case)
    ax.legend(title=r'$\dot\psi_{navi}$ [deg/turn]')
    ax.ticklabel_format(style='sci',scilimits=(0,0),useMathText=True, axis='y')
    return disp_dict

if __name__ == '__main__':
    vals = ['1','2']#,'5']
    pows = ['-2','-1']#,'0','1']
    case_names = np.array([[x+'e'+y for x in vals] for y in pows]).flatten()
    disp_dict = main(case_names)
