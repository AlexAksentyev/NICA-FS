import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import DAVEC

DIR = '../data/BYPASS_SEX_CLEAR/optimize-EB-sweep/fine-mesh/'
TAG = 'no-SEXT'

def load_parameters(path, tag):
    return np.loadtxt(path+'LATTICE-PARAMETERS:'+tag+'.txt', dtype = list(zip(['SGF1','SGF2','SGD','EBE'],[float]*4)))
def load_nu(path, tag):
    return DAVEC(path+'NU:'+tag+'.da')


EBE = np.zeros(1001)
NU0 = np.zeros(1001)
for i in range(1,1002):
    parsi = load_parameters(DIR, TAG+str(i))
    nui = load_nu(DIR, TAG+str(i))
    EBE[i-1] = parsi['EBE']
    NU0[i-1] = nui.const

plt.plot(EBE, NU0)
plt.xlabel('EBE [kV/cm]')
plt.ylabel(r'$\nu_0$')
