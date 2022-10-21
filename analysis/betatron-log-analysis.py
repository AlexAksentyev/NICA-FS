import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import pandas as pds
from sklearn import linear_model
from mpl_toolkits import mplot3d
from scipy.optimize import curve_fit

def norm(vec):
    return np.sqrt(np.sum(np.square(vec)))
def normalize(vec):
    norma = norm(vec)
    zeros = np.zeros(len(vec))
    if norma < 1e-10:
        return zeros
    return np.divide(vec, norm(vec), out=zeros)
def proj(v1, v2):
    norm2 = norm(v2)
    return v1.dot(v2)/norm2

def deltaC(ex, ey, Qx, Qy, a0, a1, deltam):
    term1 = -2*np.pi*(ex*Qx + ey*Qy)
    term2 = deltam*(a0 + a1*deltam)
    return term1 + term2

def deltaeq(a0, a1, y0, deltam, ex, ey, Qx, Qy):
    ftr1 = y0**2/(y0**2*a0 - 1)
    ftr2 = 0.5*(a1 - a0/y0**2 + 1/y0**4)
    return ftr1 * (ftr2 * deltam**2 + deltaC(ex, ey, Qx, Qy, a0, a1, deltam))
    

## CONSTANTS
ALPHA0X = -1.17578
ALPHA0Y = +2.40443
GAMMA0 = 1.1279235
EX = 3e-3 # emittance [m*rad]
EY = 3e-3 #           [m*rad]
DELTAM = 1e-4 # amplitude of synchrotron oscillations
## OPTIMUMS
# spin-chromaticity
SGopt = np.array([-0.0003492, 0.0030593, -0.0003640])
BGopt = np.array([ 0.0027942, 0.0051272, -0.0078698])
def opt_spin_sub(log):
    sub1 = log[ abs( log.SGF1 - -0.0003492)<5e-4]
    sub2 = sub1[abs(sub1.SGF2 -  0.0030593)<5e-4]
    sub3 = sub2[abs(sub2.SGD  - -0.0003640)<5e-4]
    return sub3
# beta chromaticities
def opt_beta_sub(log):
    sub1 = log[ abs( log.SGF1 -  0.0027942)<5e-4]
    sub2 = sub1[abs(sub1.SGF2 -  0.0051272)<5e-4]
    sub3 = sub2[abs(sub2.SGD  - -0.0078698)<5e-4]
    return sub3

def nearness(a, b):
    return np.sqrt(np.sum(np.square(r-point),axis=1))

def grad_nearness(point, log):
    r = np.array(list(zip(log.SGF1, log.SGF2, log.SGD)))
    return np.sqrt(np.sum(np.square(r-point),axis=1))

def st_chrom(log):
    qKx, qKy, qKd = log.qKx, log.qKy, log.qKd

## loading data
path = path = '../data/BYPASS_SEX_CLEAR/optimize-BETATUNES-gradsweep/BETATRON-LOG-SWEEP:FS-EBE.dat'

log = np.loadtxt(path,dtype=list(zip(['SGF1','SGF2','SGD','EBE','qKx','qKy','qKd','Qx','Qy', 'a1x','a1y'],[float]*11)), skiprows=1)
log = pds.DataFrame(log)

a1x, a1y, Qx, Qy = [log[e] for e in ('a1x','a1y','Qx','Qy')]

## computing ANALYTICS
deltaC_x = deltaC(EX, EY, Qx, Qy, ALPHA0X, a1x, DELTAM)
deltaC_y = deltaC(EX, EY, Qx, Qy, ALPHA0Y, a1y, DELTAM)
deltaeq_x = deltaeq(ALPHA0X, a1x, GAMMA0, DELTAM, EX, EY, Qx, Qy)
deltaeq_y = deltaeq(ALPHA0Y, a1y, GAMMA0, DELTAM, EX, EY, Qx, Qy)
log.insert(11, "deltaC_x", deltaC_x)
log.insert(12, "deltaX_y", deltaC_y)
log.insert(13, "deltaeq_x", deltaeq_x)
log.insert(14, "deltaeq_y", deltaeq_y)

## multiple linear reression on sextupole grads
regr = linear_model.LinearRegression()
X = log[['SGF1','SGF2','SGD']]
vec = {}
for e in ['qKx','qKy','qKd','Qx','Qy','a1x']:
    y = log[e]
    regr.fit(X,y)
    vec.update({e: regr.coef_})
vecn = {}
for n, v in vec.items():
    vecn.update({n: normalize(v)})

# plotting regression vectors
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot([0, vecn['qKx'][0]],[0, vecn['qKx'][1]], [0, vecn['qKd'][2]], '->k', label='qKx')
ax3.plot([0, vecn['qKy'][0]],[0, vecn['qKy'][1]], [0, vecn['qKy'][2]], '->r', label='qKy')
ax3.plot([0, vecn['qKd'][0]],[0, vecn['qKd'][1]], [0, vecn['qKd'][2]], '->b', label='qKd')
ax3.plot([0, vecn['Qx'][0]], [0, vecn['Qx'][1]],  [0, vecn['Qx'][2]],  '->g', label='Qx')
ax3.plot([0, vecn['Qy'][0]], [0, vecn['Qy'][1]],  [0, vecn['Qy'][2]],  '->m', label='Qy')
ax3.plot([0, 0],[0, 0], [0, 0], '*r')
ax3.set_xlim([-1,1])
ax3.set_ylim([-1,1])
ax3.set_zlim([-1,1])
ax3.set_xlabel('SGF1')
ax3.set_ylabel('SGF2')
ax3.set_zlabel('SGD')
plt.legend()
