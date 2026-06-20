# This file provides the code to run a standard AFIR molecular network generation using the xTB level of theory
# note! When running on sherlock, make sure CREST and xtb are installed on my sherlock directory

import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
import easyxtb

def get_induced_energy(geometry, fragments, gamma=146.44, p=6, sigma=1.0061, R_0=3.8164):
    """Gets the induced energy for a two-fragment AFIR system
    Args: 
        geometry: easyxtb geometry | the total geometry
        fragments: list{RDKit Mol} | a list of the fragments in this geometry
        gamma: float | collision energy parameter (kJ/mol) | default = 146.44 kJ/mol (35 kcal/mol)
        p: int | default = 6
        sigma: float | default = 1.0061 kJ/mol
        R_0: float | default = 3.8164 Angstrom
    Returns:
        induced_energy: float | induced energy
    """
    # check for fragment number
    if len(fragments) != 2:
        print("Error! Must be a two-fragment system!")
        return 0
    
    # make the alpha parameter
    alpha = gamma/(R_0*(2**(-1/p)-(1+(1+gamma/sigma)**(0.5))**(-1/p)))
    
    # prepare the position and radii matrices
    pos_1 = get_frag_positions(fragments[0], geometry)
    pos_2 = get_frag_positions(fragments[1], geometry)
    rad_1 = get_frag_radii(fragments[0])
    rad_2 = get_frag_radii(fragments[1])

    # calculate distance matrix (r_ij)
    diff = pos_1[:, np.newaxis, :] - pos_2[np.newaxis, :, :] # shape (N,M,3)
    r_ij = np.linalg.norm(diff, axis=2) # shape (N, M)
    
    # calculate radii matrix (R_ij)
    R_ij_sum = rad_1[:, np.newaxis] + rad_2[np.newaxis, :]

    # compute numerator and denominator (small epsilon to handle division by zero)
    epsilon = 1e-10
    scaling_matrix = (R_ij_sum/(r_ij+epsilon))**p
    num = np.sum(scaling_matrix * r_ij)
    denom = np.sum(scaling_matrix)

    # calculate induced energy
    induced_energy = alpha*num/denom
    return induced_energy

def generate_starting_confs(geometry):
    """Rotates the entire molecule in 3D space.
    Args: 
        geometry: easyxtb geometry | starting molecule
    Returns:
        rotated_geom: easyxtb geometry | rotated molecule    
    """
    return None

def get_frag_radii(fragment):
    """Provides an array of atomic radii corresponding to a specific fragment
    Args:
        fragment: RDKit Mol | an individual molecule
    Returns:
        radii: array (fragment.NumAtoms()) | an array of atomic radii in the fragment
    """
    VDW_RADII = {
    'H': 1.20,
    'C': 1.70,
    'N': 1.55,
    'O': 1.52,
    'F': 1.47,
    'P': 1.80,
    'S': 1.80,
    'Cl': 1.75,
    'Br': 1.85
    }
    radii_pre = [VDW_RADII[atom.GetSymbol()] for atom in fragment.GetAtoms()]
    radii = np.array(radii_pre)
    return radii

def get_frag_positions(fragment, geometry):
    """Returns a matrix of atomic positions corresponding to a specific fragment
    Args:
        fragment: RDKit Mol | an individual molecule
        geometry: easyxtb geometry | the larger reactive collection
    Returns:
        positions: matrix (fragment.GetNumAtoms(), 3) | a matrix corresponding to the x,y,z coordinates of each atom in fragment
    """
    indices = [atom.GetIntProp("orig_idx") for atom in fragment.GetAtoms()]
    positions = (np.array([[geometry.atoms[idx].x, geometry.atoms[idx].y, geometry.atoms[idx].z] for idx in indices]))
    return positions

def get_fragments(geometry):
    """Takes in an easyxtb geometry and finds the fragments contained in it.
    Args:
        geometry: easyxtb molecular geometry
    Returns:
        fragments: list{Mols} | list of RDKit Mols for each fragment
    """
    # make empty editable molecular graph in RDKit
    mol = Chem.RWMol()

    # add atoms from the geometry to the graph
    atom_indices = []
    for atom in geometry:
        rdkit_atom = Chem.Atom(atom.element)
        idx = mol.AddAtom(rdkit_atom)
        atom_indices.append(idx)

    # RDKit creates an editable Conformer object to map out the grid
    num_atoms = sum(1 for _ in geometry)
    conf = Chem.Conformer(num_atoms)
    for i, atom in enumerate(geometry):
        conf.SetAtomPosition(i, (atom.x, atom.y, atom.z))

    # save the atomic indices
    for idx, atom in enumerate(mol.GetAtoms()):
        atom.SetIntProp("orig_idx", idx)

    # temporary placeholder molecule to get bonds
    temp_mol = mol.GetMol()
    temp_mol.AddConformer(conf)

    # determine the bonds and split into fragments
    rdDetermineBonds.DetermineConnectivity(temp_mol)
    rdDetermineBonds.DetermineBondOrders(temp_mol)
    fragments = Chem.GetMolFrags(temp_mol, asMols=True, sanitizeFrags=False)

    return fragments

def make_smiles(mol):
    """Creates a SMILES string list from a easyxtb geometry.
    Args:
        mol: RDKit Mol | molecule
    Returns:
        smiles: str | canonical SMILES string for the molecule (Hs removed)
    """
    # remove Hs
    clean_mol = Chem.RemoveHs(mol)

    # generate SMILES
    smiles = Chem.MolToSmiles(clean_mol, canonical=True)
    return smiles

# TEST
def create_test_molecule(save_path):
    """Writes a molecule.xyz file in save_path/molecules
    Args:
        save_path: str | working directory
    """
    # make a demo methanol molecule
    methanol_xyz_content = """6
Methanol baseline test
C          0.04610        0.02340       -0.39520
O         -0.05310       -0.01540        1.01160
H         -0.45780        0.93240       -0.75110
H         -0.48280       -0.85240       -0.78520
H          1.08860        0.05540       -0.72590
H          0.44310       -0.84060        1.33230
"""
    
    # prepare directory
    target_dir = save_path / "molecules" 
    file_path = target_dir / "methanol.xyz"

    # see if it needs to be saved
    if file_path.exists():
        return

    # save the molecule
    target_dir.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(methanol_xyz_content)

# TEST
def test_get_fragments_basic_sn2(save_path):
    """Testing get_fragments function on methanol molecule.
    Result: passed
    """
    input_geom = easyxtb.Geometry.load_file(save_path / "molecules" / "basic_sn2.xyz")
    geom = easyxtb.calculate.optimize(input_geom, level="normal")
    expected = 2
    frags = get_fragments(geom) # should return a single fragment
    if len(frags) == expected:
        smiles_1 = make_smiles(frags[0])
        smiles_2 = make_smiles(frags[1])
        if  smiles_1 == "CCBr":
            if  smiles_2 == "N":
                return True
        else:
            print(f"Expected frag: CCBr and N")
            print(f"Returned: {smiles_1} and {smiles_2}")
            return False
    print(f"Expected Length: {2}")
    print(f"Returned Length: {len(frags)}")
    return False

# TEST
def test_get_fragments_methanol(save_path):
    """Testing get_fragments function on methanol molecule.
    Result: passed
    """
    input_geom = easyxtb.Geometry.load_file(save_path / "molecules" / "methanol.xyz")
    geom = easyxtb.calculate.optimize(input_geom, level="normal")
    expected = 1
    frags = get_fragments(geom) # should return a single fragment
    if len(frags) == expected:
        smiles = make_smiles(frags[0])
        if  smiles == "CO":
            return True
        else:
            print(f"Expected frag: CO")
            print(f"Returned: {smiles}")
            return False
    print(f"Expected Length: {1}")
    print(f"Returned Length: {len(frags)}")
    return False

# TEST
def test_get_frag_positions_methanol(save_path):
    """Testing get_frag_positions function on methanol molecule.
    Result: passed
    """
    input_geom = easyxtb.Geometry.load_file(save_path / "molecules" / "methanol.xyz")
    geom = easyxtb.calculate.optimize(input_geom, level="normal")
    mol = get_fragments(geom)[0] # should return a single fragment

    # test
    expected = np.array([
        [0.0461,  0.0234, -0.3952],
        [-0.0531, -0.0154,  1.0116],
        [-0.4578,  0.9324, -0.7511],
        [-0.4828, -0.8524, -0.7852],
        [ 1.0886,  0.0554, -0.7259],
        [ 0.4431, -0.8406,  1.3323]
    ])
    return np.array_equal(get_frag_positions(mol, geom), expected)

# TEST
def test_smiles_methanol(save_path):
    """Testing make_smiles function on methanol molecule.
    Result: passed
    """
    input_geom = easyxtb.Geometry.load_file(save_path / "molecules" / "methanol.xyz")
    geom = easyxtb.calculate.optimize(input_geom, level="normal")
    expected = "CO"
    frags = get_fragments(geom) # should return a single fragment
    result = make_smiles(frags[0])
    if result == expected:
        return True
    print(f"Expected {expected}")
    print(f"Returned {result}")
    return False

# TEST
def test_get_frag_radii_methanol(save_path):
    """Testing get_frag_radii function on methanol molecule.
    Result: passed
    """
    input_geom = easyxtb.Geometry.load_file(save_path / "molecules" / "methanol.xyz")
    geom = easyxtb.calculate.optimize(input_geom, level="normal")
    mol = get_fragments(geom)[0] # should return a single fragment

    # test
    expected = np.array([1.7, 1.52, 1.2, 1.2, 1.2, 1.2])
    return np.array_equal(get_frag_radii(mol), expected)

# TEST
def test_get_induced_energy_basic_sn2(save_path):
    """Testing get_induced_energy function on basic_sn2 molecules.
    Result: passed
    """
    input_geom = easyxtb.Geometry.load_file(save_path / "molecules" / "basic_sn2.xyz")
    geom = easyxtb.calculate.optimize(input_geom, level="normal")
    frags = get_fragments(geom) # should return a single fragment

    # test
    expected = 1568.8144040556358
    result = get_induced_energy(geom, frags)
    return (result == expected)

if __name__ == "__main__":
    # establish directory and save paths
    script_dir = Path(__file__).resolve().parent
    save_path = script_dir.parent / "experimental"
    create_test_molecule(save_path)

    # Tests
    print("Test Results:")
    print(f"   test_get_induced_energy_basic_sn2: {test_get_induced_energy_basic_sn2(save_path)}")
    print(f"   test_get_fragments_basic_sn2: {test_get_fragments_basic_sn2(save_path)}")
    print(f"   test_get_frag_radii_methanol: {test_get_frag_radii_methanol(save_path)}")
    print(f"   test_get_frag_positions_methanol: {test_get_frag_positions_methanol(save_path)}")
    print(f"   test_smiles_methanol: {test_smiles_methanol(save_path)}")
    print(f"   test_get_fragments_methanol: {test_get_fragments_methanol(save_path)}")

    # Finished!
    print("Complete")