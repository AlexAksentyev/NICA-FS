import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, DAVEC, load_data, Polarization
import os
from analysis2ST import plot_seq, plot_spin
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

PSVARS = list(zip(['X','A','Y','B','T','D'],[float]*6))
SEQMAP = { #indexes of cornerstone elements (in COSY indexing, SEQFULL.fox file [i.e., no RF (which is at index 0 anyway)])
    'SPD1':21,  'ARC1s':43, 'ARC1f': 236,
    'MDP1':257, 'MPD2':293, # straight section
    'ARC2s':318, 'ARC2f':511, 'SPD2':530
    }

LATTICE = '2ST+'
navipsi = 30
NTURN = '50'
ROOT = '../data/'+LATTICE+'/yGVARY/'
datadir = lambda momentum: ROOT+str(momentum)+'/SEQ/'+NTURN


def main(varyvar, spin_psi=0):
    dir_ = datadir(varyvar)
    folder  = dir_+'/NAVIPSI-{:d}/'.format(navipsi)
    print(folder)
    navi_psi_rad = np.deg2rad(180-navipsi) # 180 b/c the navigators set psi in the SPD as 180 - psi
    axis = [np.sin(navi_psi_rad), 0, np.cos(navi_psi_rad)]
    print(axis)
    ## data loading
    dat = load_data(folder, 'TRPRAY:PSI0spin-{:d}.dat'.format(spin_psi))
    spdat = load_data(folder, 'TRPSPI:PSI0spin-{:d}.dat'.format(spin_psi))
    P = Polarization.on_axis(spdat, axis)
    ## computations
    vvl = str(varyvar) + '__' + str(spin_psi) # "varyvar label"
    P.plot(1)
    plt.savefig(folder+vvl+'-pol.png', bbox_inches='tight', pad_inches=.1)
    fig, ax = plot_seq(dat, spdat,itn=(0,1))
    plt.savefig(folder+vvl+'-plots-1turn.png', bbox_inches='tight', pad_inches=.1)
    fig, ax = plot_seq(dat, spdat,itn=(0,50))
    plt.savefig(folder+vvl+'-plots.png', bbox_inches='tight', pad_inches=.1)
    plt.close('all')
    return P, dat, spdat

def spin_plot_circ(spdat, itn=(0,1), pid=[1,2,3]):
    if type(itn)==int:
        sp1 = spdat[spdat[:,0]['iteration']<itn+1]
        eid = sp1['EID'][:, pid] if itn<2 else np.arange(sp1['EID'].max()*itn+1)
    else:
        itrow = spdat[:,0]['iteration']
        ii = np.logical_and(itrow>itn[0], itrow<itn[1]+1)
        itrng = itn[1]-itn[0]
        sp1 = spdat[ii]
        eid_max = sp1['EID'].max()
        eid = eid_max*itn[0] + np.arange(eid_max*itrng)
    s0 = sp1[0,0]
    fig, ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(sp1[:,pid]['S_Z'], sp1[:,pid]['S_X'],'.')
    ax[0].set_ylabel('X')
    ax[0].plot([0, s0['S_Z']],[0, s0['S_X']], '->r', markevery=[1])
    ax[1].plot(sp1[:,pid]['S_Z'], sp1[:,pid]['S_Y'],'.')
    ax[1].set_xlabel('Z'); ax[1].set_ylabel('Y')
    ax[1].plot([0, s0['S_Z']],[0, s0['S_Y']], '->r', markevery=[1])
    for i in range(2):
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
    P = {}
    for casevar in caserng:
        P.update({casevar: main(casevar, 33)})
