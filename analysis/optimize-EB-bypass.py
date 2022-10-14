import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import load_data, DAVEC


DIR = '../data/BYPASS_SEX_CLEAR/'

GAMMA = 1.1279235
BETA = np.sqrt(GAMMA**2-1)/GAMMA # injection at y = 1.129 (242.01975 MeV) Deuterons
CLIGHT = 3e8
Lacc = 503.04
v = CLIGHT*BETA
TAU = Lacc/v

def plot_spin(spdat, rng=slice(0,-1,50),pid = [1,2,3], title=''):
    fig, ax = plt.subplots(3,1,sharex=True)
    ax[0].set_title(title)
    ax[0].plot(spdat[rng, pid]['iteration']/1000, spdat[rng,pid]['S_X'])
    ax[0].set_ylabel('S_X')
    ax[1].plot(spdat[rng, pid]['iteration']/1000, spdat[rng,pid]['S_Y'])
    ax[1].set_ylabel('S_Y')
    ax[2].plot(spdat[rng, pid]['iteration']/1000, spdat[rng,pid]['S_Z'])
    ax[2].set_xlabel('turn [x1000]'); ax[2].set_ylabel('S_Z')
    return fig, ax

if __name__ == '__main__':
    TAG = 'optimize-EB'
    
    nu_pre  = DAVEC(DIR+"NU:pre-{}.da".format(TAG)); nu_pre0 = nu_pre.const
    nu_post = DAVEC(DIR+"NU:post-{}.da".format(TAG)); nu_post0 = nu_post.const
    spdat_pre  = load_data(DIR, "TRPSPI:pre-{}.dat".format(TAG))
    spdat_post = load_data(DIR, "TRPSPI:post-{}.dat".format(TAG))
    plot_spin(spdat_pre); plot_spin(spdat_post)
