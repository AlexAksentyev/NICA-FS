import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR

DATADIR = 'data/BETA-FUNCTION/'

ELNAMES = np.load('nica_element_names.npy')
ELNAMES = np.array([e+' ['+ str(i+1) + ']' for i,e in enumerate(ELNAMES)])
D_TYPE = [('EL', int)] + list(zip(['B1RE', 'B1IM', 'B2RE','B2IM'], [float]*4))


beta = np.loadtxt(HOMEDIR+DATADIR+'BETA.dat', dtype=D_TYPE)

fig, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(beta['EL'], beta['B1RE'], '-b',  label=r'$\Re(\beta_x)$')
ax[0].plot(beta['EL'], beta['B1IM'], '--r', label=r'$\Im(\beta_x)$')
ax[0].set_ylabel(r'$\beta_x$')
ax[1].plot(beta['EL'], beta['B2RE'], '-b',  label=r'$\Re(\beta_y)$')
ax[1].plot(beta['EL'], beta['B2IM'], '--r', label=r'$\Im(\beta_y)$')
ax[1].set_ylabel(r'$\beta_y$')
for i in range(2):
    ax[i].legend()
plt.xticks(ticks=beta['EL'], labels=ELNAMES[beta['EL']-1], rotation=60)
