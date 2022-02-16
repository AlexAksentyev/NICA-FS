import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, DAVEC, load_data, Polarization
import os
from analysis2ST import plot, plot_spin, load_nbar

PSVARS = list(zip(['X','A','Y','B','T','D'],[float]*6))

LATTICE = '2ST+'
navipsi = 30
NTURN = '30000'
ROOT = '../data/'+LATTICE+'/yGVARY/'
datadir = lambda momentum: ROOT+str(momentum)+'/'+NTURN

navi_psi_rad = np.deg2rad(navipsi)
POLAXIS = [np.sin(navi_psi_rad), 0, np.cos(navi_psi_rad)]

def load_tss(folder, spin_psi=0):
    mu = DAVEC(folder+'MU:PSI0spin-'+str(spin_psi))
    nbar = load_nbar(folder, '-'+str(spin_psi))
    return mu, nbar

def main(varyvar, spin_psi=0):
    dir_ = datadir(varyvar)
    folder  = dir_+'/NAVIPSI-{:d}/'.format(navipsi)
    print(folder)
    ## data loading
    dat = load_data(folder, 'TRPRAY:PSI0spin-{:d}.dat'.format(spin_psi))
    spdat = load_data(folder, 'TRPSPI:PSI0spin-{:d}.dat'.format(spin_psi))
    P = Polarization.on_axis(spdat, POLAXIS)
    ## computations
    vvl = str(varyvar) + '__' + str(spin_psi) # "varyvar label"
    P.plot(1)
    plt.savefig(folder+vvl+'-pol.png', bbox_inches='tight', pad_inches=.1)
    Px = Polarization.on_axis(spdat[1:-1:3], POLAXIS)
    Pd = Polarization.on_axis(spdat[3:-1:3], POLAXIS)
    Px.plot(1)
    plt.savefig(folder+vvl+'-pol-X-bunch.png', bbox_inches='tight', pad_inches=.1)
    Pd.plot(1)
    plt.savefig(folder+vvl+'-pol-D-bunch.png', bbox_inches='tight', pad_inches=.1)
    fig, ax = plot(dat, spdat)
    plt.savefig(folder+vvl+'-plots.png', bbox_inches='tight', pad_inches=.1)
    fig2, ax2 = plot_spin(spdat)
    plt.savefig(folder+vvl+'-spin.png', bbox_inches='tight', pad_inches=.1)
    plt.close('all')
    return P, dat, spdat

def spin_plot_circ(spdat, pid=[1,2,3]):
    s0 = spdat[0,0]
    fig, ax = plt.subplots(3,1)
    ## z-x
    ax[0].set_xlabel('Z');  ax[0].set_ylabel('X')
    ax[0].plot(spdat[:,pid]['S_Z'], spdat[:,pid]['S_X'],'.')
    ax[0].plot([0, s0['S_Z']], [0, s0['S_X']], '->r', markevery=[1])
    ax[0].plot([0, POLAXIS[2]], [0, POLAXIS[0]], '--k')
    ## z-y
    ax[1].set_xlabel('Z'); ax[1].set_ylabel('Y')
    ax[1].plot(spdat[:,pid]['S_Z'], spdat[:,pid]['S_Y'],'.')
    ax[1].plot([0, s0['S_Z']],[0, s0['S_Y']], '->r', markevery=[1])
    ax[1].plot([0, POLAXIS[2]], [0, POLAXIS[1]], '--k')
    ## x-y
    ax[2].set_xlabel('X'); ax[2].set_ylabel('Y')
    ax[2].plot(spdat[:,pid]['S_X'], spdat[:,pid]['S_Y'],'.')
    for i in range(3):
        ax[i].grid()
    return fig, ax


if __name__ == '__main__':
    caserng = os.listdir(ROOT);
    try:
        caserng.remove('.DS_Store');
        caserng.remove('img');
    except:
        pass
    caserng = [int(x) for x in caserng]; caserng.sort()
    caserng=[2]
    P = {}
    for casevar in caserng:
        try:
            P.update({casevar: main(casevar, 33)})
        except:
            pass
                
