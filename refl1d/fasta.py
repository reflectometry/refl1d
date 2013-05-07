"""
AMINO ACIDS	

Using amino acid residues, and [1] for volumes.

For unmodified protein need to add 2*(H/D) and O for terminations.

Assumes that proteins were created in an environment with the usual H/D isotope
ratio on the non-swappable hydrogens.

[1] Perkins, Modern Physical Methods in Biochemistry Part B, 143-265 (1988)

"""

import periodictable as pt

class BaseMolecule(object):
    def __init__(self, code, name, formula, cell_volume=None, density=None):
        if formula[-1] == '-':
            charge = -1
            formula = formula[:-1]
        elif formula[-1] == '+':
            charge = +1
            formula = formula[:-1]
        else:
            charge = 0

        # Build the formula with and without deuteration assuming the base formula
        # has all solvent substitutable hydrogens replaced with deuterium.
        D = pt.formula(formula, natural_density=density)
        if cell_volume != None:
            D.density = 1e24*D.molecular_mass/cell_volume
            #print name, D.molecular_mass, cell_volume, D.density
        else:
            cell_volume = 1e24*D.molecular_mass/D.density

        self.code = code
        self.name = name
        self.formula = isotope_substitution(D, pt.D, pt.H)
        self.deuterated = D
        self.hydrogenated = isotope_substitution(D, pt.D, pt.H[1])
        self.cell_volume = cell_volume
        self.charge = charge

class AminoAcid(BaseMolecule): pass
class NucleicAcid(BaseMolecule): pass

class Nucleotide(BaseMolecule):
    def __init__(self, code, name, nucleic_acids):
        # Compute average over possible nucleotides, assuming equal weight
        n = len(nucleic_acids)
        D, cell_volume = pt.formula(), 0
        for c in nucleic_acids:
            D += (1./n)*NUCLEIC_ACIDS[c].deuterated
            cell_volume += (1./n)*NUCLEIC_ACIDS[c].cell_volume
        D = pt.formula(D)
        D.density = 1e24*D.molecular_mass/cell_volume if cell_volume else 0

        self.code = code
        self.name = name
        self.nucleic_acids = nucleic_acids
        self.formula = isotope_substitution(D, pt.D, pt.H)
        self.deuterated = D
        self.hydrogenated = isotope_substitution(D, pt.D, pt.H[1])
        self.cell_volume = cell_volume

class Sequence(object):
    """
    Convert FASTA sequence into chemical formula.

    *name* sequence name
    """
    @classmethod
    def loadall(self, filename):
        """
        Iterate over sequences in FASTA file, loading each in turn.

        Yields one FASTA sequence each cycle.
        """
        with open(filename, 'rt') as fh:
            for name, seq in read_fasta(fh):
                yield Sequence(name, seq)

    @classmethod
    def load(self, filename):
        """
        Load the first FASTA sequence from a file.
        """
        with open(filename, 'rt') as fh:
            name, seq = read_fasta(fh).next()
            return Sequence(name, seq)

    def __init__(self, name, sequence, type='faa'):
        codes = AMINO_ACID_CODES if type == 'faa' else NUCLEIC_ACID_CODES
        parts = tuple(codes[c] for c in sequence)
        cell_volume = sum(p.cell_volume for p in parts)
        charge = sum(p.charge for p in parts)
        structure = []
        for p in parts: structure.extend(list(p.deuterated.structure))
        D = pt.formula(structure)
        D = pt.formula(D.atoms)  # compact formula
        D.density = 1e24*D.molecular_mass / cell_volume

        self.name = name
        self.sequence = sequence
        self.formula = isotope_substitution(D, pt.D, pt.H)
        self.deuterated = D
        self.hydrogenated = isotope_substitution(D, pt.D, pt.H[1])
        self.cell_volume = cell_volume
        self.charge = charge

def read_fasta(fp):
    """
    Iterate over the sequences in a FASTA file.

    Each iteration is a pair (sequence name, sequence codes).
    """
    name, seq = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith(">"):
            if name: yield (name, ''.join(seq))
            name, seq = line, []
        else:
            seq.append(line)
    if name: yield (name, ''.join(seq))


def isotope_substitution(formula, source, target, portion=1):
    """
    Substitute one atom/isotope in a formula with another in some proportion.

    *formula* is the formula being updated.

    *source* is the isotope/element to be substituted.

    *target* is the replacement isotope/element.
 
    *portion* is the proportion of source which is substituted for target.
    """
    atoms = formula.atoms
    if source in atoms:
        mass = formula.mass
        mass_reduction = atoms[source]*portion*(source.mass - target.mass)
        density = formula.density * (mass - mass_reduction)/mass
        atoms[target] = atoms.get(target,0) + atoms[source]*portion
        if portion == 1:
            del atoms[source]
        else:
            atoms[source] *= 1-portion
    else:
        density = formula.density
    return pt.formula(atoms, density=density)

# FASTA code table
_ = lambda code, V, formula, name: AminoAcid(code, name, formula, cell_volume=V)
AMINO_ACID_CODES = dict((a.code,a) for a in (
#code, volume, formula,        name
_("A",  91.5, "C3H4DNO",    "alanine"),
_("C", 105.6, "C3H3DNOS",   "cysteine"),
_("D", 124.5, "C4H3DNO3-",  "aspartic acid"),
_("E", 155.1, "C5H5DNO3-",  "glutamic acid"),
_("F", 203.4, "C9H8DNO",    "phenylalanine"),
_("G",  66.4, "C2H2DNO",    "glycine"),
_("H", 167.3, "C6H5D3N3O+", "histidine"),
_("I", 168.8, "C6H10DNO",   "isoleucine"),
_("K", 171.3, "C6H9D4N2O+", "lysine"),
_("L", 168.8, "C6H10DNO",   "leucine"),
_("M", 170.8, "C5H8DNOS",   "methionine"),
_("N", 135.2, "C4H3D3N2O2", "asparagine"),
_("P", 129.3, "C5H7NO",     "proline"),
_("Q", 161.1, "C5H5D3N2O2", "glutamine"),
_("R", 202.1, "C6H7D6N4O+", "arginine"),
_("S",  99.1, "C3H3D2NO2",  "serine"),
_("T", 122.1, "C4H5D2NO2",  "threonine"),
_("V", 141.7, "C5H8DNO",    "valine"),
_("W", 237.6, "C11H8D2N2O", "tryptophan"),
_("Y", 203.6, "C9H7D2NO2",  "tyrosine"),
))

_ = lambda code, r, formula, name: NucleicAcid(code, name, formula, density=r)
NUCLEIC_ACIDS = dict((a.code,a) for a in (
_("A",   1.6,  "C5H5N5",   "adenine"),
_("C",   1.55, "C4H5N3O",  "cytosine"),
_("G",   2.2,  "C5H5NO5",  "guanine"),
_("T",   1.23, "C5H6N2O2", "thymine"),
_("U",   1.32, "C4H4N2O2", "uracil"),
))

_ = lambda code, acids, name: Nucleotide(code, name, acids)
NUCLEIC_ACID_CODES = dict((a.code,a) for a in (
#code, nucleotides,  name
_("A", "A",     "adenosine"),
_("C", "C",     "cytidine"),
_("G", "G",     "guanosine"),
_("T", "T",     "thymidine"),
_("U", "U",     "uridine"),
_("R", "AG",    "purine"),
_("Y", "CTU",   "pyrimidine"),
_("K", "GTU",   "ketone"),
_("M", "AC",    "amino"),
_("S", "CG",    "strong"),
_("W", "ATU",   "weak"),
_("B", "CGTU",  "not A"),
_("D", "AGTU",  "not C"),
_("H", "ACTU",  "not G"),
_("V", "ACG",   "not T"),
_("N", "ACGTU", "any base"),
_("X", "",      "masked"),
_("-", "",      "gap"),
))
NUCLEIC_ACID_CODES['-'] = NUCLEIC_ACID_CODES['N']


def fasta_table():
    rows = [v for k,v in sorted(AMINO_ACID_CODES.items())]
    rows += [Sequence("beta casein",beta_casein)]

    print "%20s %7s %7s %7s %5s %5s %5s %5s %5s"%(
        "name","M(H2O)","M(D2O)","volume","den","#el","xray","nH2O","nD2O")
    for v in rows:
        electrons = sum(num*el.number for el,num in v.formula.atoms.items()) - v.charge
        nHsld = pt.neutron_sld(v.hydrogenated, wavelength=4.75)
        nDsld = pt.neutron_sld(v.deuterated, wavelength=4.75)
        Xsld = pt.xray_sld(v.formula, wavelength=pt.Cu.K_alpha)
        print "%20s %7.1f %7.1f %7.1f %5.2f %5d %5.2f %5.2f %5.2f"%(
            v.name, v.hydrogenated.mass, v.deuterated.mass, v.cell_volume,
            v.formula.density,
            electrons, Xsld[0], nHsld[0], nDsld[0]) 

beta_casein = "RELEELNVPGEIVESLSSSEESITRINKKIEKFQSEEQQQTEDELQDKIHPFAQTQSLVYPFPGPIPNSLPQNIPPLTQTPVVVPPFLQPEVMGVSKVKEAMAPKHKEMPFPKYPVEPFTESQSLTLTDVENLHLPLPLLQSWMHQPHQPLPPTVMFPPQSVLSLSQSKVLPVPQKAVPYPQRDMPIQAFLLYQEPVLGPVRGPFPIIV"

def test():
    # Beta casein results checked against Duncan McGillivray's spreadsheet
    # beta casein 23561.9 23880.9 30872.9  1.27 12614 11.55  1.68  2.75
    s = Sequence("beta casein", beta_casein)
    assert abs(s.deuterated.mass-23880.9) < 0.1
    assert abs(s.formula.density-1.27) < 0.01
    assert abs(pt.neutron_sld(s.deuterated, wavelength=4.75)[0]-2.75) < 0.01

if __name__=="__main__":
    fasta_table()
