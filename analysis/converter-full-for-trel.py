import re

HOME = '/Users/alexaksentyev/REPOS/NICA-FS/'
INFILE = 'madx-scripts/ST-mode/NICA4SOL.seq'
OUTFILE = 'src/setups/SECOND-ST/SEQFULL.fox'
NUM_ACC = 1 # current array position of the element to be written

SELECT_ELEMENTS = []  # elements to be swapped for DL

el_dict = {
    'MONITOR': ('DL', 1),
    'DRIFT': ('DL', 1),
    'QUADRUPOLE': ('QUAD', 3),
    'SEXTUPOLE': ('SEXT', 3),
    'OCTUPOLE': ('OCT', 3),
    'SBEND': ('SBEND', 9),
    'RBEND': ('RBEND', 3),
    'SOLENOID': ('SOLENOID',2),
    'MULTIPOLE': ('MULT', 3),
    'KICKER': ('KICK', 3),
    'RFCAVITY': ('RFCAV', 4)
    }

var_dclr = []
var_def = []

argpos_dict = {
    'SEXT': {'TILT': 1, 'KNL':2},
    'SBEND': {'L':1, 'ANGLE':2, 'TILT':3, 'E1':4,'H1':5, 'E2':6, 'H2':7, 'FINT':8, 'FINTX':9},
    'RBEND': {'L':1, 'ANGLE':2, 'TILT':3},
    'DL': {'L':1},
    'QUAD': {'L':1, 'TILT':2, 'K1':3, 'KNL':3},
    'SEXT': {'L':1, 'TILT':2, 'KNL':3},
    'OCT':  {'L':1, 'TILT':2, 'KNL':3},
    'MULT': {'TILT':1, 'ARR':2, 'NARR':3},
    'KICK': {'L':1, 'HKICK':2, 'VKICK':3},
    'SOLENOID': {'L':1, 'KS':2},
    'RFCAV': {'L':1, 'VOLT':2, 'LAG':3, 'HARMON':4}
    }

lbl_dict = {}

def write_header(fhandle):
    fhandle.seek(0,0)
    for i, decl in enumerate(var_dclr):
        fhandle.write(decl)
    fhandle.write('\n\n')
    for i, dfn in enumerate(var_def):
        fhandle.write(dfn)
    fhandle.write('\n\n\n')
    fhandle.write('{________________________________________________________________________________}\n')
    fhandle.write('\n\n\n')
    

def insert_args(element):
    elem_name, n_arg = el_dict[element[0]]
    n_zeros = n_arg-len(element[1:])
    elem = [element[0]]
    elem.extend(['name=0' for i in range(n_zeros)])
    for par in element[1:]:
        par_name, par_val = par.strip().split('=')
        elem.insert(argpos_dict[elem_name][par_name],par_name+'='+par_val)
    return elem

def identify_mult(lbl, element):
    regexp = '\{[\d\D]+\}'
    # check if it is a multipole
    if len(re.findall("MULTIPOLE", element))>0:
        arr_str = re.findall(regexp, element)[0]
        arr = arr_str.replace(" ","").strip('\{').strip('\}').split(',')
        arr_name = lbl+'MARR'
        narr_name = 'N'+arr_name
        # change the array of values to the last value (all others are 0 anyway)
        element = re.sub('KNL='+arr_str, 'ARR={}, NARR={}'.format(arr_name, narr_name), element)
        # create variable declarations
        var_dclr.append('VARIABLE {0} 1 {1}; VARIABLE {2} 1;\n'.format(arr_name, len(arr), narr_name))
        for i, val in enumerate(arr): # create array variable definition
            var_def.append('{}({}) := {};\n'.format(arr_name, i+1, val))
        var_def.append('{} := {};\n'.format(narr_name, i+1))
    else:
        pass
    return element

def parse_element(element):
    elem = element.strip(';\n').strip().split(',')
    elem = insert_args(elem)
    return elem

def form_string(element):
    out_str = el_dict[element[0]][0] + ' ' # element line
    # arguments
    for par in element[1:]:
        par_val = par.replace(' ', '').split('=')[1]
        if (par_val.find('+',1)>0 or par_val.find('-',1)>0):
            par_val = '('+par_val+')'
        out_str += par_val+' '
    # finish
    out_str += ';' 
    return out_str

def swap_for_DL(element):
    out_str = 'DL '
    for par in element[1:]:
        par_name, par_val = par.replace(' ', '').split('=')
        if par_name == 'L':
            out_str += par_val
    return out_str + ' ;'

def write_dict(line):
    try:
        line, _ = line.split("!") # ignore inline comments
    except:
        pass
    print(line)
    lbl, elem =  re.sub(":=", "=", " ".join(line.split())).split(":")
    print('**', lbl+': '+elem)
    elem = identify_mult(lbl, elem)
    elem = parse_element(elem)
    out_string = swap_for_DL(elem) if elem[0] in SELECT_ELEMENTS else form_string(elem)
    out_string += ' {' + lbl + '}\n' # adding comment to procedure string
    lbl_dict.update({lbl : out_string}) # filling the label dictionary
    

def write_file(line, fout):
    global NUM_ACC
    seq = "".join(line.split())[:-1] # remove trailing comma
    seq = seq.strip(');')
    seq = seq.split(',')
    for idx, element in enumerate(seq):            
        out_line = 'UM; '
        out_line += lbl_dict[element]
        out_line += '\t SMAPS {} MAPARR SPNRARR;\n'.format(NUM_ACC + idx)
        fout.write(out_line)
    NUM_ACC += idx + 1

fout = open(HOME+OUTFILE,'w')
with open(HOME+INFILE, 'r') as fin:
    for cnt, line in enumerate(fin):
        if cnt<2: #
            pass
        elif cnt<548: # 544 for First
            print(cnt)
            write_dict(line)
        elif cnt<550: # 546 for First
            print('++', cnt)
            print(line)
            pass
        elif cnt>549: # 545 for First
            print('##',cnt)
            print(line)
            if (line[0]!='\n' and line[0]!='/'):
                write_file(line, fout)
            else:
                pass

fout.close()

with open(HOME+OUTFILE,'r+') as fout:
    content = fout.read()
    write_header(fout)
    fout.write(content)
    

