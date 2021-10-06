import numpy as np

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
