import numpy as np
import matplotlib.pyplot as plt; plt.ion()


c = 2.99792458e3 # m/s
beta = .463920811
E0 = 100 #kV/cm
B = lambda E, beta: E*1e5/(c*beta)/1e4

B0 = B(E0, beta)

barr = np.linspace(.4,.5,100)
plt.plot(barr, B(E0, barr), label='B0 = {:4.2f} [kGs]'.format(B0))
plt.axvline(x=beta,color='r')
plt.grid()
plt.legend()
plt.xlabel(r'$\beta$')
plt.ylabel('B[kGs]')
