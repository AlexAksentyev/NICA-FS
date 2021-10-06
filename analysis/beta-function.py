import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR

DATADIR = 'data/BETA-FUNCTION/FIRST-ST/SEQFULL/'

ELNAMES = np.load('nica_element_names.npy')
ELNAMES = np.array([e+' ['+ str(i+1) + ']' for i,e in enumerate(ELNAMES)])
D_TYPE = [('EL', int)] + list(zip(['1RE', '1IM', '2RE','2IM'], [float]*4))


beta = np.loadtxt(HOMEDIR+DATADIR+'BETA.dat', dtype=D_TYPE)
mu = np.loadtxt(HOMEDIR+DATADIR+'MU.dat', dtype=D_TYPE)

fig, ax1 = plt.subplots(3,1,sharex=True)
ax1[0].plot(beta['EL'], beta['1RE'], '-b',  label=r'$\Re(\beta_x)$')
ax1[0].plot(beta['EL'], beta['1IM'], '--r', label=r'$\Im(\beta_x)$')
ax1[0].set_ylabel(r'$\beta_x$')
ax1[1].plot(beta['EL'], beta['2RE'], '-b',  label=r'$\Re(\beta_y)$')
ax1[1].plot(beta['EL'], beta['2IM'], '--r', label=r'$\Im(\beta_y)$')
ax1[1].set_ylabel(r'$\beta_y$')
ax1[2].plot(beta['EL'], beta['1RE'], '-b',  label=r'$\Re(\beta_x)$')
ax1[2].plot(beta['EL'], beta['2RE'], '-r',  label=r'$\Re(\beta_y)$')
ax1[2].set_ylabel(r'$\beta$')
for i in range(3):
    ax1[i].legend()
#plt.xticks(ticks=beta['EL'], labels=ELNAMES[beta['EL']-1], rotation=60)

fig, ax2 = plt.subplots(2,1,sharex=True)
ax2[0].plot(mu['EL'], mu['1RE'], '-b',  label=r'$\Re(\mu_x)$')
ax2[0].plot(mu['EL'], mu['1IM'], '--r', label=r'$\Im(\mu_x)$')
ax2[0].set_ylabel(r'$\mu_x$')
ax2[1].plot(mu['EL'], mu['2RE'], '-b',  label=r'$\Re(\mu_y)$')
ax2[1].plot(mu['EL'], mu['2IM'], '--r', label=r'$\Im(\mu_y)$')
ax2[1].set_ylabel(r'$\mu_y$')
for i in range(2):
    ax2[i].legend()
#plt.xticks(ticks=mu['EL'], labels=ELNAMES[mu['EL']-1], rotation=60)
