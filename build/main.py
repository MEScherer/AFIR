# note! When running on sherlock, make sure CREST and xtb are installed on my sherlock directory

import numpy as np
from pathlib import Path

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
    with open(file_path, "w") as f:
        f.write(methanol_xyz_content)

if __name__ == "__main__":
    save_path = Path("results")
    create_test_molecule(save_path)
    print("Complete")