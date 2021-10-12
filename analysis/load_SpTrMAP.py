import numpy as np
import matplotlib.pyplot as plt; plt.ion()

def load_SpTrMAP(fname):
    data = np.genfromtxt(fname, dtype=[('X', float),('Y', float),('Z', float),('EXP', object)],
                             comments='----------', skip_footer=1,
                             delimiter=(1,20,20,20,7),
                             usecols=range(1,5),
                             converters={4: lambda x: x.strip()})
    ii = data['EXP']==b'000000'
    SpTrM0 = data[ii]
    SpTrM = np.zeros((3,3))
    for i in range(3):
        SpTrM[i] = list(SpTrM0[i])[:3]
    return SpTrM

def rot_angle(matrix):
    tr = matrix.trace()
    angle = np.arccos((tr-1)/2)
    return angle

def euler_angles(matrix):
    ax = np.arctan2( matrix[2,1],  matrix[2,2])
    ay = np.arctan2(-matrix[2,0],  np.sqrt(matrix[2,1]**2 + matrix[2,2]**2))
    az = np.arctan2( matrix[1,0],  matrix[0,0])
    return ax, ay, az

if __name__ == '__main__':
    fname = lambda nrg, num: '../data/ELEMENT_MAPS/SECOND-ST/{}MeV/ELEM_{}-SPIN'.format(nrg, num)
    nrg1 = 130;  gamma1 = 1 + nrg1/938
    nrg2 = 3059; gamma2 = 1 + nrg2/938
    gamrat = gamma2/gamma1

    tot_elnum = 546; eid = np.arange(1, tot_elnum+1)
    ANG1 = np.recarray(tot_elnum, dtype=[('X',float),('Y',float),('Z',float)])
    ANG2 = np.recarray(tot_elnum, dtype=[('X',float),('Y',float),('Z',float)])
    for i in range(546):
        tmp1 = load_SpTrMAP(fname(130,  i+1))
        tmp2 = load_SpTrMAP(fname(3059, i+1))
        tmp1 = np.rad2deg(euler_angles(tmp1))
        tmp2 = np.rad2deg(euler_angles(tmp2))
        ANG1[i] = tmp1[0], tmp1[1], tmp1[2]
        ANG2[i] = tmp2[0], tmp2[1], tmp2[2]

    fig, ax = plt.subplots(3,1,sharex=True)
    lbl = ['X','Y','Z']
    ax[0].plot(eid, ANG2['X'], label=nrg2)
    ax[1].plot(eid, ANG2['Y'], label=nrg2)
    ax[2].plot(eid, ANG2['Z'], label=nrg2)
    ax[0].plot(eid, ANG1['X'], label=nrg1)
    ax[1].plot(eid, ANG1['Y'], label=nrg1)
    ax[2].plot(eid, ANG1['Z'], label=nrg1)
    for i in range(3):
        ax[i].legend()
        ax[i].set_ylabel(lbl[i])

