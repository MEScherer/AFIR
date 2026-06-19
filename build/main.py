# This file provides the code to run a standard AFIR molecular network generation using the xTB level of theory
# note! When running on sherlock, make sure CREST and xtb are installed on my sherlock directory

import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
import easyxtb

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

def make_smiles(geometry):
    """Creates a SMILES string from a easyxtb geometry.
    Args:
        geometry: easyxtb molecular geometry
    Returns:
        smiles: str | canonical SMILES string
    """
    # make empty editable molecular graph in RDKit
    mol = Chem.RWMol()

    # add atoms from the geometry to the graph
    atom_indices = []
    for atom in geometry:
        rdkit_atom = Chem.Atom(atom.element)
        idx = mol.AddAtom(rdkit_atom)
        atom_indices.append(idx)

    # use distance thresholds to calculate and assign bonds
    # RDKit creates an editable Conformer object to map out the grid
    num_atoms = sum(1 for _ in geometry)
    conf = Chem.Conformer(num_atoms)
    for i, atom in enumerate(geometry):
        conf.SetAtomPosition(i, (atom.x, atom.y, atom.z))

    # temporary placeholder molecule to get bonds
    temp_mol = mol.GetMol()
    temp_mol.AddConformer(conf)
    rdDetermineBonds.DetermineConnectivity(temp_mol)
    rdDetermineBonds.DetermineBondOrders(temp_mol)
    clean_mol = Chem.RemoveHs(temp_mol)

    # generate SMILES
    smiles = Chem.MolToSmiles(clean_mol, canonical=True)
    return smiles

# TEST
def test_smiles_methanol(save_path):
    """Testing make_smiles function on methanol molecule.
    Result: passed
    """
    input_geom = easyxtb.Geometry.load_file(save_path / "molecules" / "methanol.xyz")
    geom = easyxtb.calculate.optimize(input_geom, level="normal")
    expected = "CO"
    result = make_smiles(geom)
    if result == expected:
        return True
    print(f"Expected {expected}")
    print(f"Returned {result}")
    return False

if __name__ == "__main__":
    # establish directory and save paths
    script_dir = Path(__file__).resolve().parent
    save_path = script_dir.parent / "results"
    create_test_molecule(save_path)

    # Tests
    print("Test Results:")
    print(f"   test_smiles_methanol: {test_smiles_methanol(save_path)}")

    # Finished!
    print("Complete")