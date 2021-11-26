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
    step = int(dat[1,0]['TURN'])
    if True:                                                               # nbar plot
        fig1, ax1 = plt.subplots(3,1,sharex=True)
        ax1[0].set_title(r'$\dot \psi = $ {} [deg/{}-turn]'.format(case_name,step))
        ax1[2].set_xlabel('TURN #')
        for i, var in enumerate(['X', 'Y', 'Z']):
            ax1[i].plot(muarr['TURN'], muarr['N'+var], '-')
            ax1[i].set_ylabel(r'$\bar n_{}$'.format(var))
    if True:                                                        # spin-vector plot
        fig3, ax3 = plt.subplots(3,1,sharex=True)
        ax3[0].set_title(r'$\dot \psi = $ {} [deg/{}-turn]'.format(case_name,step))
        ax3[2].set_xlabel('TURN #')
        for i, var in enumerate(['X', 'Y', 'Z']):
            ax3[i].plot(spdat[:,1:]['TURN'], spdat[:,1:]['S_'+var], '--')
            ax3[i].plot(s0['TURN'], s0['S_'+var], '-k')
            ax3[i].set_ylabel(r'$\vec s_{}$'.format(var))
    if True:                                                        # phase space plot
        fig2, ax2 = plt.subplots(3,1)
        ax2[0].set_title(r'$\dot \psi = $ {} [deg/{:d}-turn]'.format(case_name,step))
        ax2[0].plot(dat[:,:3]['X']*1e3, dat[:,:3]['A']*1e3)
        ax2[0].set_xlabel('X [mm]'); ax2[0].set_ylabel('A [mrad]')
        ax2[1].plot(dat[:,:3]['Y']*1e3, dat[:,:3]['B']*1e3)
        ax2[1].set_xlabel('Y [mm]'); ax2[1].set_ylabel('B [mrad]')
        ax2[2].plot(dat[:,:3]['T']*1e3, dat[:,:3]['D'])
        ax2[2].set_xlabel('T [mm]'); ax2[2].set_ylabel('D [unit]')
        for i in range(3):
            ax2[i].ticklabel_format(style='sci',scilimits=(0,0),useMathText=True)
    return disp, dat[:,0]['TURN']

def main(case_names):
    disp_dict = {}
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('TURN #')
    ax.set_ylabel(r'$\sigma[\cos(\vec s, \vec s_0)]$')
    for i, case in enumerate(case_names):
        disp, nturn = process(case)
        disp_dict.update({case:disp})
        ax.plot(nturn, disp, label=case)
    ax.legend(title=r'$\dot\psi_{navi}$ [deg/turn]')
    ax.ticklabel_format(style='sci',scilimits=(0,0),useMathText=True, axis='y')
    return disp_dict

if __name__ == '__main__':
    vals = ['5']
    pows = ['-1', '0']
    case_names = np.array([[x+'e'+y for x in vals] for y in pows]).flatten()
    disp_dict = main(case_names)
