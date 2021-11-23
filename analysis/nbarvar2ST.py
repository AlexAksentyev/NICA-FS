import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import load_data, Polarization

LATTICE = 'SECOND-ST'
ENERGY = '130' # MeV


def main(case, spin_psi='=PSInavi'):
    folder = '../data/'+LATTICE+'/'+ENERGY+'MeV/NAVI-VARI/RATE_'+case+'/'
    dat = load_data(folder, 'TRPRAY:PSI0spin{}.dat'.format(spin_psi))
    spdat = load_data(folder, 'TRPSPI:PSI0spin{}.dat'.format(spin_psi))
    muarr = load_data(folder, 'TRMUARR:PSI0spin{}.dat'.format(spin_psi))
    rate = np.diff(np.rad2deg(np.arcsin(muarr['NY'][:,0])))
    print('psi <rate-of-change> = ', rate.mean(), '[deg/turn]')
    return dat, spdat, muarr

if __name__ == '__main__':
    dat, spdat, muarr = main('1e-2')
