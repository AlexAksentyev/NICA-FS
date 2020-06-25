import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR, Data, TAU, Polarization
import os

def plot_TD(case, pid=1, num_seg=5):
    fig, ax = plt.subplots(1,1)
    N = int(case['iteration'].shape[0]/num_seg)
    for i in range(num_seg):
        rng = slice(i*N, (i+1)*N)
        sec0, sec1 = (case['iteration'][j*N,0]*TAU for j in [i,i+1])
        ax.plot(case['T'][rng,pid]*1e3, case['D'][rng,pid]*1e2, '.',
                    label='{:3.1f}--{:3.1f} sec'.format(sec0, sec1)
                    )
    ax.set_xlabel(r'$\ell$ [mm]')
    ax.set_ylabel(r'$\delta$ [%]')
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText=True)
    ax.legend()
    return fig, ax

if __name__ == '__main__':
    path = lambda case: HOMEDIR + 'data/REPORT/DEUTERON/BENDS24/24MTURN/Y-bunch/SPD-{}/'.format(case)
    CASES = ['0+', '90+', '90-']
    
    sp = {case: Data(path(case), 'TRPSPI:MAIN.dat') for case in CASES}
    ps = {case: Data(path(case), 'TRPRAY:MAIN.dat') for case in CASES}
    pol = {case: Polarization.on_axis(sp[case], [0,1,0]) for case in CASES}

    for case in CASES:
        print(case)
        d = path(case)
        if not os.path.exists(d+'img/'):
            os.makedirs(d+'img/')
        fig, ax = plot_TD(ps[case][3000:],num_seg=4)
        plt.savefig(d+'img/PS_TD_{}.png'.format(case), dpi=450, bbox_inches='tight', pad_inches=.1)
        plt.close()
