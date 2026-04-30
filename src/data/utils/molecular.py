from typing import List
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as bt

RDLogger.DisableLog('rdApp.*')

import numpy as np


def read_molecules(filepath: str, sanitize: bool=False, remove_hydrogens: bool=False):
    """ Read molecules from a file.
    """

    ############  molecules file  ############
    if filepath.endswith('.sdf'):

        suppl = Chem.SDMolSupplier(
            filepath,
            removeHs=remove_hydrogens,
            sanitize=sanitize
        )

    #############  smiles file  ##############
    elif filepath.endswith('.smi'):
        
        suppl = Chem.SmilesSupplier(
            filepath,
            removeHs=remove_hydrogens,
            sanitize=sanitize
        )

    else:
        raise NotImplementedError(
            f'Molecules supplier for file {filepath} not implemented'
        )

    return suppl


def kekulize_molecule(mol):
    """ Kekulize a molecule.
    """

    Chem.Kekulize(mol)
    return mol



def get_molecule_stats(mols: List[Chem.Mol]):
    """Function for computing general statistics on set of molecules.
    Currently returns:
    - number of atoms: avg, std, and total
    - number of bonds: avg, std, and total
    - a list with all found atom types
    - a list with all found bond types

    Parameters
    ----------
    mols : 
    """
    
    atoms = set()
    bonds = set()
    l_num_atoms = []
    l_num_bonds = []

    for mol in mols:
        l_num_atoms.append(mol.GetNumAtoms())
        l_num_bonds.append(mol.GetNumBonds())

        for atom in mol.GetAtoms():
            atoms.add(atom.GetSymbol())

        for bond in mol.GetBonds():
            bonds.add(str(bond.GetBondType()))

    atoms = list(atoms)
    bonds = list(bonds)

    ret_dict = {
        'num_atoms_avg': np.mean(l_num_atoms).item(),
        'num_atoms_std': np.std(l_num_atoms).item(),
        'num_atoms_total': np.sum(l_num_atoms).item(),
        'num_bonds_avg': np.mean(l_num_bonds).item(),
        'num_bonds_std': np.std(l_num_bonds).item(),
        'num_bonds_total': np.sum(l_num_bonds).item(),
        'atom_types': atoms,
        'bond_types': bonds
    }

    return ret_dict