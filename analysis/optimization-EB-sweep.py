import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import DAVEC

DIR = '../data/BYPASS_SEX_CLEAR/optimize-EB-sweep/'
TAG = 'opt-SEXT-sext'

def load_parameters(path, tag):
    return np.loadtxt(path+'LATTICE-PARAMETERS:'+tag+'.txt', dtype = list(zip(['SGF1','SGF2','SGD','EBE'],[float]*4)))
def load_nu(path, tag):
    return DAVEC(path+'NU:'+tag+'.da')


EBE = np.zeros(41)
NU0 = np.zeros(41)
for i in range(1,42):
    parsi = load_parameters(DIR, TAG+str(i))
    nui = load_nu(DIR, TAG+str(i))
    EBE[i-1] = parsi['EBE']
    NU0[i-1] = nui.const

plt.plot(EBE, NU0)
plt.xlabel('EBE [kV/cm]')
plt.ylabel(r'$\nu_0$')
