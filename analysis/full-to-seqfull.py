

HOME = '/Users/alexaksentyev/REPOS/NICA-FS/'
FULLFILE = 'src/setups/BYPASS/FULL_SEX_wRC.fox'
SEQFILE = 'src/setups/BYPASS/SEQFULL_SEX_wRC.fox'

STARTLINE = 62
ENDLINE = 597


def write_file(in_line, where, accpos):
    out_line = 'UM; '
    out_line += in_line
    out_line += ' SMAPS {} MAPARR SPNRARR;\n'.format(accpos)
    where.write(out_line)

fout = open(HOME+SEQFILE, 'w')
with open(HOME+FULLFILE, 'r') as fin:
    accpos = 1 # start filling MAPARR, SPNR from index 1
    for cnt, line in enumerate(fin):
        if cnt < STARTLINE-1: # skip the first lines
            pass
        elif cnt < ENDLINE:   # write the sequence lines (excluding RF)
            write_file(line, fout, accpos)
            accpos += 1 # move position
        else:                 # skip the rest of file
            pass
fout.close()
