# Artificial Force Induced Reaction (AFIR) Prototype

An automated reaction path search engine prototype built in Python. This tool implements the AFIR method to discover chemical transition states and reaction pathways using a file-based pipeline coupled with `easyxtb`.

## Features
- **Automated Fragment Pushing:** Distorts the Potential Energy Surface (PES) using a distance-dependent artificial force term based on atomic covalent radii.
- **xTB Integration:** Leverages tight-binding semi-empirical quantum mechanics (`easyxtb` backend) for ultra-fast gradient evaluations.
- **Avogadro Compatible:** Designed to ingest standard `.xyz` molecular geometries exported directly from Avogadro 2.

## Installation & Dependencies
Ensure you have Python 3.8+ and the following packages installed:
```bash
pip install numpy easyxtb
