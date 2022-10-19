import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import HOMEDIR

def prep_elnames(lattice, rf=True):
    elnames = list(np.load('../src/setups/'+lattice+'/FULL.npy'))
    if rf:
        elnames.insert(0, 'RF')
    elnames = np.array(elnames)
    elnames = np.array([e+' ['+ str(i+1) + ']' for i,e in enumerate(elnames)]) # add the ordinal
    return elnames

def prep_elnames2(raw=False):
    elnames =  list(np.load('../src/setups/BYPASS/FULL_SEX_CLEAR-element_names.npy', allow_pickle=True))
    elnames.insert(0, 'RF')
    elnames = np.array(elnames)
    elnames = np.array([e.strip('\n').strip('\{').strip('\}') for e in elnames])
    if not raw:
        elnames = np.array([e +' ['+ str(i+1) + ']' for i,e in enumerate(elnames)])
    return elnames
    
LATTICE = 'BYPASS'
DATADIR = 'data/BETA-FUNCTION/'+LATTICE+'/SEQFULL/'
## adapted to SEX_CLEAR lattice
ELNAMES = prep_elnames2()
ELEMENTS_RAW = prep_elnames2(True)
iSF1 = ELEMENTS_RAW == 'SF1'
iSF2 = ELEMENTS_RAW == 'SF2'
iSD = ELEMENTS_RAW == 'SD'
iSEX = iSF1 + iSF2 + iSD
##
TICK_STP = 1 #20
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
    ax1[i].grid()
#plt.xticks(ticks=beta['EL'][::TICK_STP], labels=ELNAMES[beta['EL'][::TICK_STP]-1], rotation=90)
plt.xticks(ticks=np.arange(len(ELNAMES))[iSEX], labels=ELNAMES[iSEX], rotation=90)

fig, ax2 = plt.subplots(2,1,sharex=True)
ax2[0].plot(mu['EL'], mu['1RE'], '-b',  label=r'$\Re(\mu_x)$')
ax2[0].plot(mu['EL'], mu['1IM'], '--r', label=r'$\Im(\mu_x)$')
ax2[0].set_ylabel(r'$\mu_x$')
ax2[1].plot(mu['EL'], mu['2RE'], '-b',  label=r'$\Re(\mu_y)$')
ax2[1].plot(mu['EL'], mu['2IM'], '--r', label=r'$\Im(\mu_y)$')
ax2[1].set_ylabel(r'$\mu_y$')
for i in range(2):
    ax2[i].legend()
#plt.xticks(ticks=mu['EL'][::TICK_STP], labels=ELNAMES[mu['EL'][::TICK_STP]-1], rotation=90)
plt.xticks(ticks=np.arange(len(ELNAMES))[iSEX], labels=ELNAMES[iSEX], rotation=90)
