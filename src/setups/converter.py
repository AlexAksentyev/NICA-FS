import numpy as np

elname_map = {
    'QUADRUPOLE': 'QUAD',
    'SEXTUPOLE' : 'SEXT',
    'DIPOLE'    : 'DL',
    'WIEN'      : 'WIEN',
    'DL'        : 'DL'
    }

arg_map = {
    'QUADRUPOLE': lambda arr: arr[0]+'/100 '+arr[2]+' '+arr[1]+'*10/CONS(CHIM)',
    'SEXTUPOLE' : lambda arr: arr[0]+' '+arr[2]+' '+arr[1]+'*1000',
    'DIPOLE'    : lambda arr: arr[0]+'/100',
    'WIEN'      : lambda arr: ' '.join(arr),
    'DL'        : lambda arr: ' '.join(arr)
    }

lines = []
with open('BNL-lattice.txt') as fin:
    for line in fin:
        try:
            elem, comment = line.strip().split(';')
            elem = elem.split()
            out_string = elname_map[elem[0]] + ' ' + arg_map[elem[0]](elem[1:]) + '; '
            out_string += comment + '\n'
        except:
            out_string = line
            print(line)
        lines.append(out_string)

with open('BNL-lattice-conv.fox', 'a') as fout:
    for line in lines:
        fout.write(line)
