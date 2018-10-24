"""
Biomolecule support.

:class:`Molecule` lets you define biomolecules with labile hydrogen atoms
specified using tritium (T) in the chemical formula.  The biomolecule object
creates forms with natural isotope ratio, all hydrogen and all deuterium.
Density can be provided as natural density or cell volume.  A %D2O contrast
match value is computed for matching the molecule SLD in the presence of
labile hydrogens.  :method:`Molecule.D2Osld` computes the neutron SLD for
the solvated molecule in a %D2O solvent.

:fun:`D2Omatch` computes the %D2O constrast match value given the fully
hydrogenated and fully deuterated forms.

:class:`Sequence` lets you read amino acid and DNA/RNA sequences from FASTA
files.

Tables for common molecules are provided[1]:

    *AMINO_ACID_CODES* : amino acids indexed by FASTA code

    *RNA_CODES*, DNA_CODES* : nucleic bases indexed by FASTA code

    *RNA_BASES*, DNA_BASES* : individual nucleic acid bases

    *NUCLEIC_ACID_COMPONENTS*, *LIPIDS*, *CARBOHYDRATE_RESIDUES*

Neutron SLD for water at 20C is also provided as *H2O_SLD* and *D2O_SLD*.

For unmodified protein need to add 2*T and O for terminations.

Assumes that proteins were created in an environment with the usual H/D isotope
ratio on the non-swappable hydrogens.

[1] Perkins, Modern Physical Methods in Biochemistry Part B, 143-265 (1988)

"""
from __future__ import division, print_function

import periodictable as pt

class Molecule(object):
    """
    Specify a biomolecule by name, chemical formula, cell volume and charge.

    Labile hydrogen positions should be coded using tritium (T) rather than H.  That
    way the tritium can be changed to H[1] for solutions with pure water, H for solutions
    with a natural abundance of water or D for solutions with pure deuterium.

    Attributes
    ==========

    *formula* is the original tritiated formula.  You can create the hydrogenated or
    deuterated forms using::

        from periodictable import H, D, T, fast
        hydrogenated = isotope_substitution
    :fun:`isotope_substitution(M.formula, ` with *formula*, T
    and periodictable.H or periodictable.D.

    *D2Omatch* is the % D2O in H2O required to contrast match the molecule, including
    the the proton swapping effect.

    *natural*/*H*/*D* are the formulae with tritium replaced by naturally occurring D/H ratio,
    by pure H and by pure D respectively.  *sld*/*H_sld*/*D_sld* are the corresponding slds.
    *mass* and *density* are available as *H.mass* and *H.density*, etc.

    *charge* is the charge on the molecule

    *cell_volume* is the estimated cell volume for the molecule
    """
    def __init__(self, name, formula, cell_volume=None, natural_density=None, charge=0):
        M = pt.formula(formula, natural_density=natural_density)

        # Fill in density or cell_volume
        if cell_volume is not None:
            M.density = 1e24*M.molecular_mass/cell_volume if cell_volume > 0 else 0
        else:
            cell_volume = 1e24*M.molecular_mass/M.density

        self.cell_volume = cell_volume
        self.name = name
        self.formula = M
        self.natural = isotope_substitution(M, pt.T, pt.H)
        self.H = isotope_substitution(M, pt.T, pt.H[1])
        self.D = isotope_substitution(M, pt.T, pt.D)

        self.sld = pt.neutron_sld(self.natural, wavelength=5)[0]
        self.H_sld = pt.neutron_sld(self.H, wavelength=5)[0]
        self.D_sld = pt.neutron_sld(self.D, wavelength=5)[0]

        self.D2Omatch = D2Omatch(self.H_sld, self.D_sld)
        self.charge = charge

    def D2Osld(self, volume_fraction=1., D2O_fraction=0.):
        """
        Neutron SLD of the molecule in a %D2O solvent.
        """
        solvent_sld = D2O_fraction*D2O_SLD + (1-D2O_fraction)*H2O_SLD
        solute_sld = D2O_fraction*self.D_sld + (1-D2O_fraction)*self.H_sld
        return volume_fraction*solute_sld + (1-volume_fraction)*solvent_sld

class Sequence(Molecule):
    """
    Convert FASTA sequence into chemical formula.

    *name* sequence name
    *sequence* code string
    *type* aa|dna|rna
       aa: amino acid sequence
       dna: dna sequence
       rna: rna sequence

    Note: rna sequence files treat T as U and dna sequence files treat U as T.
    """
    @staticmethod
    def loadall(filename):
        """
        Iterate over sequences in FASTA file, loading each in turn.

        Yields one FASTA sequence each cycle.
        """
        with open(filename, 'rt') as fh:
            for name, seq in read_fasta(fh):
                yield Sequence(name, seq)

    @staticmethod
    def load(filename):
        """
        Load the first FASTA sequence from a file.
        """
        with open(filename, 'rt') as fh:
            name, seq = next(read_fasta(fh))
            return Sequence(name, seq)

    def __init__(self, name, sequence, type='aa'):

        codes = _CODE_TABLES[type]
        parts = tuple(codes[c] for c in sequence)
        cell_volume = sum(p.cell_volume for p in parts)
        charge = sum(p.charge for p in parts)
        structure = []
        for p in parts:
            structure.extend(list(p.formula.structure))
        formula = pt.formula(structure).hill

        Molecule.__init__(self, name, formula,
                          cell_volume=cell_volume, charge=charge)
        self.sequence = sequence

# Water density at 20C; neutron wavelength doesn't matter (use 5 A).
H2O_SLD = pt.neutron_sld(pt.formula("H2O@0.9982"), wavelength=5)[0]
D2O_SLD = pt.neutron_sld(pt.formula("D2O@0.9982"), wavelength=5)[0]

def D2Omatch(H_sld, D_sld):
    """
    Find the D2O% concentration of solvent such that neutron SLD of the
    material matches the neutron SLD of the solvent.

    *H_sld*, *D_sld* are the SLDs for the hydrogenated and deuterated forms
    of the material respectively, where *D* includes all the labile protons
    swapped for deuterons.  Water SLD is calculated at 20 C.

    Note that the resulting percentage is only meaningful between
    0% to 100%.  Beyond 100% you will need an additional constrast agent
    in the 100% D2O solvent to increase the SLD enough to match.
    """
    # SLD(%Dsample + (1-%)Hsample) = SLD(%D2O + (1-%)H2O)
    # %SLD(Dsample) + (1-%)SLD(Hsample) = %SLD(D2O) + (1-%)SLD(H2O)
    # %(SLD(Dsample) - SLD(Hsample) + SLD(H2O) - SLD(D2O)) = SLD(H2O) - SLD(Hsample)
    # % = 100*(SLD(H2O) - SLD(Hsample)) / (SLD(Dsample) - SLD(Hsample) + SLD(H2O) - SLD(D2O))
    return 100*(H2O_SLD - H_sld) / (D_sld - H_sld + H2O_SLD - D2O_SLD)

def read_fasta(fp):
    """
    Iterate over the sequences in a FASTA file.

    Each iteration is a pair (sequence name, sequence codes).
    """
    name, seq = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith(">"):
            if name:
                yield (name, ''.join(seq))
            name, seq = line, []
        else:
            seq.append(line)
    if name:
        yield (name, ''.join(seq))


def isotope_substitution(formula, source, target, portion=1):
    """
    Substitute one atom/isotope in a formula with another in some proportion.

    *formula* is the formula being updated.

    *source* is the isotope/element to be substituted.

    *target* is the replacement isotope/element.

    *portion* is the proportion of source which is substituted for target.
    """
    # Note: formula.atoms creates a new structure each time
    atoms = formula.atoms
    if source in atoms:
        mass = formula.mass
        mass_reduction = atoms[source]*portion*(source.mass - target.mass)
        density = formula.density * (mass - mass_reduction)/mass
        atoms[target] = atoms.get(target, 0) + atoms[source]*portion
        if portion == 1:
            del atoms[source]
        else:
            atoms[source] *= 1-portion
    else:
        density = formula.density
    return pt.formula(atoms, density=density)

# FASTA code table
def _(code, V, formula, name):
    if formula[-1] == '-':
        charge = -1
        formula = formula[:-1]
    elif formula[-1] == '+':
        charge = +1
        formula = formula[:-1]
    else:
        charge = 0
    molecule = Molecule(name, formula, cell_volume=V, charge=charge)
    molecule.code = code  # Add code attribute so we can write as well as read
    return code, molecule

AMINO_ACID_CODES = dict((
    #code, volume, formula,        name
    _("A", 91.5, "C3H4TNO", "alanine"),
    _("C", 105.6, "C3H3TNOS", "cysteine"),
    _("D", 124.5, "C4H3TNO3-", "aspartic acid"),
    _("E", 155.1, "C5H5TNO3-", "glutamic acid"),
    _("F", 203.4, "C9H8TNO", "phenylalanine"),
    _("G", 66.4, "C2H2TNO", "glycine"),
    _("H", 167.3, "C6H5T3N3O+", "histidine"),
    _("I", 168.8, "C6H10TNO", "isoleucine"),
    _("K", 171.3, "C6H9T4N2O+", "lysine"),
    _("L", 168.8, "C6H10TNO", "leucine"),
    _("M", 170.8, "C5H8TNOS", "methionine"),
    _("N", 135.2, "C4H3T3N2O2", "asparagine"),
    _("P", 129.3, "C5H7NO", "proline"),
    _("Q", 161.1, "C5H5T3N2O2", "glutamine"),
    _("R", 202.1, "C6H7T6N4O+", "arginine"),
    _("S", 99.1, "C3H3T2NO2", "serine"),
    _("T", 122.1, "C4H5T2NO2", "threonine"),
    _("V", 141.7, "C5H8TNO", "valine"),
    _("W", 237.6, "C11H8T2N2O", "tryptophan"),
    _("Y", 203.6, "C9H7T2NO2", "tyrosine"),
))

def _(formula, V, name):
    molecule = Molecule(name, formula, cell_volume=V)
    return name, molecule

NUCLEIC_ACID_COMPONENTS = dict((
    # formula, volume, name
    _("NaPO3", 60, "phosphate"),
    _("C5H6TO3", 125, "ribose"),
    _("C5H7O2", 115, "deoxyribose"),
    _("C5H2T2N5", 114, "adenine"),
    _("C4H2TN2O2", 99, "uracil"),
    _("C5H4TN2O2", 126, "thymine"),
    _("C5HT3N5O", 119, "guanine"),
    _("C4H2T2N3O", 103, "cytosine"),
))

CARBOHYDRATE_RESIDUES = dict((
    # formula, volume, name
    _("C6H7T3O5", 171.9, "Glc"),
    _("C6H7T3O5", 166.8, "Gal"),
    _("C6H7T3O5", 170.8, "Man"),
    _("C6H7T4O5", 170.8, "Man (terminal)"),
    _("C8H10T3NO5", 222.0, "GlcNAc"),
    _("C8H10T3NO5", 232.9, "GalNAc"),
    _("C6H7T3O4", 160.8, "Fuc (terminal)"),
    _("C11H11T5NO8", 326.3, "NeuNac (terminal)"),
    # Glycosaminoglycans
    _("C14H15T5NO11Na", 390.7, "hyaluronate"),  # GlcA.GlcNAc
    _("C14H17T5NO13SNa", 473.5, "keratan sulphate"), # Gal.GlcNAc.SO4
    _("C14H15T4NO14SNa", 443.5, "chondroitin sulphate"), # GlcA.GalNAc.SO4
))

LIPIDS = dict((
    # formula, volume, name
    _("CH2", 27, "methylene"),
    _("CD2", 27, "methylene-D"),
    _("C10H18NO8P", 350, "phospholipid headgroup"),
    _("C6H5O6", 240, "triglyceride headgroup"),
    _("C36H72NO8P", 1089, "DMPC"),
    _("C36H20D52NO8P", 1089, "DMPC-D52"),
    _("C29H55T3NO8P", 932, "DLPE"),
    _("C27H45TO", 636, "cholesteral"),
    _("C45H78O2", 1168, "oleate"),
    _("C57H104O6", 1617, "trioleate form"),
    _("C39H77T2N2O2P", 1166, "palmitate ester"),
))

def _(code, formula, V, name):
    molecule = Molecule(name, formula, cell_volume=V)
    molecule.code = code
    return code, molecule

RNA_BASES = dict((
    # code, formula, volume, name
    _("A", "C10H8T3N5O6PNa", 299, "adenosine"),
    _("T", "C9H8T2N2O8PNa", 284, "uridine"), # Use T for U in RNA
    _("G", "C10H7T4N5O7PNa", 304, "guanosine"),
    _("C", "C9H8T3N3O7PNa", 288, "cytidine"),
))

DNA_BASES = dict((
    # code, formula, volume, %D2O matchpoint, name
    _("A", "C10H9T2N5O5PNa", 289, "adenosine"),
    _("T", "C10H11T1N2O7PNa", 301, "thymidine"),
    _("G", "C10H8T3N5O6PNa", 294, "guanosine"),
    _("C", "C9H9T2N3O6PNa", 278, "cytidine"),
))

def _nucleic_acid_average(bases, code_table):
    """
    Compute average over possible nucleotides, assuming equal weight if
    precise nucleotide is not known
    """
    n = len(bases)
    D, cell_volume = pt.formula(), 0
    for c in bases:
        D += code_table[c].formula
        cell_volume += code_table[c].cell_volume
    if n > 0:
        D, cell_volume = (1/n) * D, cell_volume/n
    return D, cell_volume

def _(code, bases, name):
    D, V = _nucleic_acid_average(bases, RNA_BASES)
    rna = Molecule(name, D.hill, cell_volume=V)
    rna.code = code
    D, V = _nucleic_acid_average(bases, DNA_BASES)
    dna = Molecule(name, D.hill, cell_volume=V)
    dna.code = code
    return (code, rna), (code, dna)

RNA_CODES, DNA_CODES = [dict(v) for v in zip(
    #code, nucleotides,  name
    _("A", "A", "adenosine"),
    _("C", "C", "cytidine"),
    _("G", "G", "guanosine"),
    _("T", "T", "thymidine"),
    _("U", "T", "uridine"), # RNA_BASES["T"] is uridine
    _("R", "AG", "purine"),
    _("Y", "CT", "pyrimidine"),
    _("K", "GT", "ketone"),
    _("M", "AC", "amino"),
    _("S", "CG", "strong"),
    _("W", "AT", "weak"),
    _("B", "CGT", "not A"),
    _("D", "AGT", "not C"),
    _("H", "ACT", "not G"),
    _("V", "ACG", "not T"),
    _("N", "ACGT", "any base"),
    _("X", "", "masked"),
    _("-", "", "gap"),
)]


_CODE_TABLES = {
    'aa': AMINO_ACID_CODES,
    'dna': DNA_CODES,
    'rna': RNA_CODES,
}

def fasta_table():
    rows = []
    rows += [v for k, v in sorted(AMINO_ACID_CODES.items())]
    rows += [v for k, v in sorted(NUCLEIC_ACID_COMPONENTS.items())]
    rows += [Sequence("beta casein", beta_casein)]

    print("%20s %7s %7s %7s %5s %5s %5s %5s %5s %5s"%(
        "name", "M(H2O)", "M(D2O)", "volume",
        "den", "#el", "xray", "nH2O", "nD2O", "%D2O match"))
    for v in rows:
        protons = sum(num*el.number for el, num in v.formula.atoms.items())
        electrons = protons - v.charge
        xray_sld = pt.xray_sld(v.formula, wavelength=pt.Cu.K_alpha)
        print("%20s %7.1f %7.1f %7.1f %5.2f %5d %5.2f %5.2f %5.2f %5.1f"%(
            v.name, v.H.mass, v.D.mass, v.cell_volume, v.natural.density,
            electrons, xray_sld[0], v.H_sld, v.D_sld, v.D2Omatch))

beta_casein = "RELEELNVPGEIVESLSSSEESITRINKKIEKFQSEEQQQTEDELQDKIHPFAQTQSLVYPFPGPIPNSLPQNIPPLTQTPVVVPPFLQPEVMGVSKVKEAMAPKHKEMPFPKYPVEPFTESQSLTLTDVENLHLPLPLLQSWMHQPHQPLPPTVMFPPQSVLSLSQSKVLPVPQKAVPYPQRDMPIQAFLLYQEPVLGPVRGPFPIIV"

def test():
    from periodictable.constants import avogadro_number
    # Beta casein results checked against Duncan McGillivray's spreadsheet
    # beta casein 23561.9 23880.9 30872.9  1.27 12614 11.55  1.68  2.75
    s = Sequence("beta casein", beta_casein)
    assert abs(s.D.mass-23880.9) < 0.1
    #print "density", s.mass/avogadro_number/s.cell_volume*1e24
    assert abs(s.natural.mass/avogadro_number/s.cell_volume*1e24 - 1.267) < 0.01
    assert abs(s.D_sld-2.75) < 0.01

    # Check that X-ray sld is independent of isotope
    H = isotope_substitution(s.formula, pt.T, pt.H)
    D = isotope_substitution(s.formula, pt.T, pt.D)
    H_sld, D_sld = pt.xray_sld(H, wavelength=1.54), pt.xray_sld(D, wavelength=1.54)
    #print H_sld, D_sld
    assert abs(H_sld[0] - D_sld[0]) < 1e-10

def main():
    import sys
    import os
    if len(sys.argv) == 1 or sys.argv[1] in ('-h', '--help', '-?'):
        print("usage: python -m periodictable.fasta file|sequence [%D]")
        sys.exit(1)
    if os.path.exists(sys.argv[1]):
        seq_list = Sequence.loadall(sys.argv[1])
    else:
        seq_list = [Sequence("", sys.argv[1])]
    if len(sys.argv) > 2:
        concentration = 0.01*float(sys.argv[2])
    else:
        concentration = 0.50

    for seq in seq_list:
        mixture_sld = seq.H_sld*(1-concentration) + seq.D_sld*concentration
        print("%s mass:%.1f H-sld: %.3f  D-sld: %.3f  D[%g%%]-sld: %.3f" %
              (seq.name, seq.natural.mass, seq.H_sld, seq.D_sld, 100*concentration, mixture_sld))

if __name__ == "__main__":
    #main()
    fasta_table()
