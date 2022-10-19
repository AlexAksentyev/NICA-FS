import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import pandas as pds
from sklearn import linear_model
from mpl_toolkits import mplot3d

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

path = path = '../data/BYPASS_SEX_CLEAR/optimize-BETATUNES-gradsweep/BETATRON-LOG-SWEEP:FS-EBE.dat'

log = np.loadtxt(path,dtype=list(zip(['SGF1','SGF2','SGD','EBE','qKx','qKy','qKd','Qx','Qy'],[float]*9)), skiprows=1)
log = pds.DataFrame(log)

sub = log[(log['SGF1']==-3e-3)&(log['SGF2']==-3e-3)]

X = log[['SGF1','SGF2','SGD']]
qKx, qKy, qKd, Qx, Qy = [log[e] for e in ('qKx','qKy','qKd','Qx','Qy')]

regr = linear_model.LinearRegression()

vec = {}
for e in ['qKx','qKy','qKd','Qx','Qy']:
    y = log[e]
    regr.fit(X,y)
    vec.update({e: regr.coef_})
vecn = {}
for n, v in vec.items():
    vecn.update({n: normalize(v)})

# plotting regression vectors
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111, projection='3d')
# ax1.plot([0, vec['qKx'][0]],[0, vec['qKx'][1]], [0, vec['qKd'][2]], '->k', label='qKx')
# ax1.plot([0, vec['qKy'][0]],[0, vec['qKy'][1]], [0, vec['qKy'][2]], '->r', label='qKy')
# ax1.plot([0, vec['qKd'][0]],[0, vec['qKd'][1]], [0, vec['qKd'][2]], '->b', label='qKd')
# ax1.plot([0, 0],[0, 0], [0, 0], '*r')
# ax1.set_xlim([-3.5,11.5])
# ax1.set_ylim([-3.5,11])
# ax1.set_zlim([-16.5,3])
# ax1.set_xlabel('SGF1')
# ax1.set_ylabel('SGF2')
# ax1.set_zlabel('SGD')
# plt.legend()
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection='3d')
# ax2.plot([0, vec['Qx'][0]], [0, vec['Qx'][1]],  [0, vec['Qx'][2]],  '->g', label='Qx')
# ax2.plot([0, vec['Qy'][0]], [0, vec['Qy'][1]],  [0, vec['Qy'][2]],  '->m', label='Qy')
# ax2.plot([0, 0],[0, 0], [0, 0], '*r')
# ax2.set_xlim([-500,2300])
# ax2.set_ylim([-500,2200])
# ax2.set_zlim([-2250,550])
# ax2.set_xlabel('SGF1')
# ax2.set_ylabel('SGF2')
# ax2.set_zlabel('SGD')
# plt.legend()
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
