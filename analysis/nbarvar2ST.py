import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from analysis import load_data, Polarization

LATTICE = 'SECOND-ST'
ENERGY = '130' # MeV
NTURN = '80'
DATDIR = '../data/'+LATTICE+'/'+ENERGY+'MeV/NAVI-VARI/'+NTURN+'/'

def main(spin_psi='=PSInavi'):
    dat = load_data(DATDIR, 'TRPRAY:PSI0spin{}.dat'.format(spin_psi))
    spdat = load_data(DATDIR, 'TRPSPI:PSI0spin{}.dat'.format(spin_psi))
    muarr = load_data(DATDIR, 'TRMUARR:PSI0spin{}.dat'.format(spin_psi))
    return dat, spdat, muarr

if __name__ == '__main__':
    dat, spdat, muarr = main()
