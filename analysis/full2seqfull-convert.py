import numpy as np
HOME = '/Users/alexaksentyev/REPOS/NICA-FS/'
INFILE =  'src/setups/BYPASS/FULL_SEX_CLEAR.fox'
OUTFILE = 'src/setups/BYPASS/SEQFULL_SEX_CLEAR-test.fox'

I1e = 53 # index of the first element in file (string number - 1)
Ile = 588 # last index
Nel = Ile-I1e + 1

fin = open(HOME+INFILE, 'r')
lines_in = fin.readlines(); fin.close()

lines_out = lines_in[I1e:Ile+1]
lengths = np.zeros(Nel, dtype=float)
tags    = np.zeros(Nel, dtype=object)

with open(HOME+OUTFILE, 'w') as fout:
    for i, string in enumerate(lines_out):
        print(string)
        split_string = string.split(' ')
        el_length, tag = split_string[1], split_string[-1]
        el_length = float(el_length) if el_length!='LEN_S' else .3
        string = 'UM; ' + string.strip('\n') + ' SMAPS {} MAPARR SPNRARR; \n'.format(i+1)
        lines_out[i] = string
        tags[i] = tag
        lengths[i] = el_length
        fout.write(string)

np.save(HOME+'src/setups/BYPASS/FULL_SEX_CLEAR-element_names', tags)

with open( HOME+'src/setups/BYPASS/FULL_SEX_CLEAR-element_lengths.fox', 'w' ) as fout2:
    for i, element_l in enumerate(lengths):
        string = 'ARRAY({}) := {}; {}'.format(i+1, element_l, tags[i]);
        fout2.write(string)

