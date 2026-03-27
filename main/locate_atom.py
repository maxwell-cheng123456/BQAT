import torch
vocab = torch.load('../vocab/vocab_with_error.pt')
# print(vocab)
def find_atom(token):
    Vocab_List={'0':'0',
            '1':'0',
            '2':'0',
            '3':'0',
            '4':'0',
            '5':'0',
            '6':'0',
            '7':'C',
            '8':'O',
            '9':'0',
            '10':'0',
            '11':'0',
            '12':'C',
            '13':'N',
            '14':'C',
            '15':'0',
            '16':'C',
            '17':'0',
            '18':'N',
            '19':'O',
            '20':'S',
            '21':'Cl',
            '22':'N',
            '23':'S',
            '24':'0',
            '25':'0',
            '26':'0',
            '27':'0',
            '28':'O',
            '29':'0',
            '30':'N',
            '31':'F',
            '32':'0',
            '33':'0',
            '34':'Br',
            '35':'B',
            '36':'N',
            '37':'P',
            '38':'Si',
            '39':'I',
            '40':'C',
            '41':'C',
            '42':'N',
            '43':'0',
            '44':'0',
            '45':'P',
            '46':'N',
            '47':'Sn',
            '48':'Mg',
            '49':'Sn',
            '50':'C',
            '51':'Cu',
            '52':'Si',
            '53':'N',
            '54':'0',
            '55':'Zn',
            '56':'0',
            '57':'S',
            '58':'Se',
            '59':'S',
            '60':'S',
            '61':'S',
            '62':'Si',
            '63':'Se',
            '64':'P',
            '65':'P',
            '66':'Zn',
            '67':'S',
            '68':'N',
            '69':'Mg',
            '70':'P',
            '71':'N',
            '72':'S',
            '73':'P',
            '74':'0',
            '75':'0',
            '76':'Si',
            '77':'N',
            '78':'N',
            '79':'0'

    }
    return Vocab_List[token]

def get_bond_energy(atom1,atom2,bond_type):
        bond_energy_list={
                'B-F': 644,
                'B-O': 515,
                'Br-Br': 193,
                'C-B': 393,
                'C-Br': 276,
                'C-C': 332,
                'C=C': 533,
                'C#C':837,
                'C$C':533,
                'C-Cl': 328,
                'C-F': 485,
                'C-H': 414,
                'C-I': 240,
                'C-N': 305,
                'C$N':483,
                'C=N':615,
                'C#N': 891,
                'C-O': 326,
                'C=O': 728,
                'C-P': 305,
                'C=P': 489,
                'C-S': 272,
                'C$S':425,
                'C=S': 536,
                'C-Si': 347,
                'Cl-Cl': 243,
                'Cs-I': 337,
                'N-H': 389,
                'N-N': 159,
                'N$N':637,
                'N#N': 946,
                'N=N':456,
                'N-O': 230,
                'N=O': 607,
                'N=S': 325,
                'Na-Br': 367,
                'Na-Cl': 412,
                'Na-F': 519,
                'Na-H': 186,
                'Na-I': 304,
                'O-H': 464,
                'O-O': 146,
                'O=O': 498,
                'P-Br': 272,
                'P-Cl': 331,
                'P-H': 322,
                'P-O': 410,
                'P=O': 632,
                'P=S': 430,
                'P-P': 213,
                'N-P':189,
                'Pb-O':382,
                'Pb-S':346,
                'Rb-Br':381,
                'Rb-Cl':428,
                'Cs-I':337,
                'F-F': 153,
                'F-S': 263,
                'H-H': 436,
                'H-Br': 366,
                'H-Cl': 431,
                'H-F': 565,
                'H-I': 298,
                'I-I': 151,
                'K-Br': 380,
                'K-Cl': 433,
                'K-F': 498,
                'K-I': 325,
                'Li-Cl': 469,
                'Li-H': 238,
                'Li-I': 345,
                'Rb-F': 494,
                'Rb-I': 319,
                'S-H': 339,
                'S-O': 364,
                'S=O': 535,
                'S-S': 184,
                'S=S': 290,
                'N-S': 312,
                'Se-H': 314,
                'Se$C': 240,
                'Se-C': 210,
                'Se-N': 180,
                'Se-Se': 232,
                'Si-Cl': 360,
                'Si-F': 552,
                'Si-H': 377,
                'Si-O': 460,
                'Si-Si': 176,
                'Si-N': 382,
                'Sn-C': 102,
                'Cl-S': 242,
                'P-S': 280,
                'C$P':489,
                'S$S':438,
                'C-Zn': 30,
                'Br-Zn': 32,
                'C-Mg': 27,
                'I-Zn': 26,
                'Cl-Sn': 41,
                'Sn-Sn': 74,
                'Br-S': 190

        }
        if bond_type==3:
                b='-'
        elif bond_type==4:
                b='#'
        elif bond_type==2:
                b='='
        elif bond_type==1:
                b='$'
        else:
                return 0
        bond1=atom1+b+atom2
        bond2=atom2+b+atom1
        try:
                try:
                        return bond_energy_list[bond1]
                except:
                        return bond_energy_list[bond2]
        except:
                if bond_type==3:
                        return 300
                else:
                        return 600
def get_bond_length(atom1,atom2,bond_type):
        bond_length_list={
                'Br-Br':229,
                'C-B':156,
                'C-Br':194,
                'C-C':154,
                'C=C':134,
                'C#C':120,
                'C-Cl':177,
                'C-F':138,
                'C-I':214,
                'C-N':148,
                'C=N':135,
                'C#N':116,
                'C-O':143,
                'C=O':120,
                'N-N':145,
                'N=N':125,
                'N#N':110,
                'N-O':146,
                'N=O':114,
                'O-O':146,
                'O=O':120,
                'P-Br':220,
                'P-Cl':203,
                'C-P':187,
                'C-S':182,
                'C=S':156,
                'C-Si':186,
                'Cl-Cl':199,
                'F-F':140,
                'I-I':266,
                'P-O':163,
                'P=O':138,
                'S-S':207,
                'C$S':169,
                'C$N':141.5,
                'C$O':131.5,
                'C$C':144

        }
        if bond_type==3:
                b='-'
        elif bond_type==4:
                b='#'
        elif bond_type==2:
                b='='
        elif bond_type==1:
                b='$'
        else:
                return 0

        bond1=atom1+b+atom2
        bond2=atom2+b+atom1
        try:
                try:
                        return bond_length_list[bond1]
                except:
                        return bond_length_list[bond2]
        except:

                if bond_type==3:#单键
                        return 179.2
                elif bond_type==2:#双键
                        return 130.2
                else:
                        return 155




if __name__=='__main__':
    a=find_atom('8')
    print(a)

    e=get_bond_length(find_atom('7'),find_atom('8'),4)#3是单键，2是双键，1是芳香建，4是三键
    print(e)