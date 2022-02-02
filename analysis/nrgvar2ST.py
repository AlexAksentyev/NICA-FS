import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, DAVEC, load_data, Polarization
import os
from analysis2ST import plot, plot_spin, load_nbar

PSVARS = list(zip(['X','A','Y','B','T','D'],[float]*6))

LATTICE = 'SECOND-ST'
NTURN = '30000'
ROOT = '../data/'+LATTICE+'/MOMVARY/'
datadir = lambda momentum: ROOT+str(momentum)+'MeV:c/'+NTURN

def load_tss(folder):
    mu = DAVEC(folder+'MU:PSI0spin-0')
    nbar = load_nbar(folder, '-0')
    return mu, nbar

def main(momentum):
    dir_ = datadir(momentum)
    folder  = dir_+'/NAVIPSI-0/'
    print(folder)
    navi_psi_rad = np.deg2rad(180-0) # 180 b/c the navigators set psi in the SPD as 180 - psi
    axis = [0, np.sin(navi_psi_rad), np.cos(navi_psi_rad)]
    print(axis)
    ## data loading
    dat = load_data(folder, 'TRPRAY:PSI0spin-0.dat')
    spdat = load_data(folder, 'TRPSPI:PSI0spin-0.dat')
    P = Polarization.on_axis(spdat, axis)
    ## computations
    mom = str(momentum)
    P.plot(1)
    plt.savefig(folder+mom+'-pol.png', bbox_inches='tight', pad_inches=.1)
    Px = Polarization.on_axis(spdat[1:-1:3], axis)
    Px.plot(1)
    plt.savefig(folder+mom+'-pol-X-bunch.png', bbox_inches='tight', pad_inches=.1)
    fig, ax = plot(dat, spdat)
    plt.savefig(folder+mom+'-plots.png', bbox_inches='tight', pad_inches=.1)
    fig2, ax2 = plot_spin(spdat)
    plt.savefig(folder+mom+'-spin.png', bbox_inches='tight', pad_inches=.1)
    plt.close('all')
    return P

def spin_dyn(momentum):
    dir_ = datadir(momentum)
    folder  = dir_+'/NAVIPSI-0/'
    spdat = load_data(folder, 'TRPSPI:PSI0spin-0.dat')
    # s0 = np.zeros(spdat.shape[0], dtype=list(zip(['X','Y','Z'],[float]*3)))
    # for lbl in ['X','Y','Z']:
    #     s0[lbl] = spdat[:,1]['S_'+lbl]
    return spdat

def _spana(P,pid1=11,pid2=14):
    fig, ax = plt.subplots(3,3,sharey='row',sharex='col')
    for i, lbl in enumerate(['X','Y','Z']):
        ax[i,0].set_ylabel('S_'+lbl)
        for j, mom in enumerate([511,3200, 4800]):
            ax[i,j].plot(P[mom][:,pid1]['S_'+lbl],  label=str(pid1+1))
            ax[i,j].plot(P[mom][:,pid2]['S_'+lbl], label=str(pid2+1))
            ax[i,j].plot(P[mom][:,0]['S_'+lbl],  label='ref')
            ax[0,j].set_title(str(mom)+' MeV/c')
        ax[i,0].legend()
    return fig, ax


if __name__ == '__main__':
    caserng = os.listdir(ROOT);
    try:
        caserng.remove('.DS_Store');
        caserng.remove('img');
    except:
        pass
    caserng = [int(x[:-5]) for x in caserng]; caserng.sort()
    P = {}
    for mom in caserng:
        P.update({mom: main(mom)})