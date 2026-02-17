#!/usr/bin/env python3
"""
Prepare PDB structure for RFDiffusion peptide design
- Remove ligands
- Identify binding pocket residues
- Clean structure

Created by: Yonglan Liu
Date: 2026-02-06
"""

from Bio import PDB
import numpy as np
import argparse

def remove_ligands(pdb_file, output_file, keep_chains=None):
    """Remove small molecule ligands and water, keep protein chains"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    io = PDB.PDBIO()
    
    class ProteinSelect(PDB.Select):
        def accept_residue(self, residue):
            # Keep only standard amino acids
            if residue.get_id()[0] == ' ':  # Standard residues have ' ' as hetfield
                return True
            return False
        
        def accept_chain(self, chain):
            if keep_chains is None:
                return True
            return chain.get_id() in keep_chains
    
    io.set_structure(structure)
    io.save(output_file, ProteinSelect())
    print(f"Cleaned structure saved to {output_file}")
    return structure

def find_ligand_binding_residues(pdb_file, ligand_specs, distance_cutoff=5.0):
    """
    Identify protein residues within distance_cutoff of ligands
    
    Args:
        pdb_file: Path to PDB file
        ligand_specs: List of tuples [(chain_id, resid), ...] for each ligand
        distance_cutoff: Distance in Angstroms
    
    Returns:
        List of binding residues in format "A123"
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)
    
    # Get all ligand atoms from all specified ligands
    ligand_atoms = []
    for ligand_chain, ligand_resid in ligand_specs:
        found = False
        for model in structure:
            for chain in model:
                if chain.get_id() == ligand_chain:
                    for residue in chain:
                        if residue.get_id()[1] == ligand_resid:
                            ligand_atoms.extend([atom for atom in residue.get_atoms()])
                            found = True
                            print(f"  Found ligand: chain {ligand_chain}, residue {ligand_resid}")
                            break
        
        if not found:
            print(f"  Warning: Ligand not found at chain {ligand_chain}, residue {ligand_resid}")
    
    if not ligand_atoms:
        print("ERROR: No ligand atoms found!")
        return []
    
    print(f"  Total ligand atoms: {len(ligand_atoms)}")
    
    # Find nearby protein residues
    binding_residues = set()  # Use set to avoid duplicates
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':  # Only protein residues
                    for atom in residue.get_atoms():
                        for lig_atom in ligand_atoms:
                            dist = atom - lig_atom  # BioPython computes distance
                            if dist < distance_cutoff:
                                res_info = f"{chain.get_id()}{residue.get_id()[1]}"
                                binding_residues.add(res_info)
                                break
    
    return sorted(list(binding_residues))

def get_chain_info(pdb_file):
    """Get information about chains in the structure"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    chain_info = {}
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            residues = [r for r in chain.get_residues() if r.get_id()[0] == ' ']
            if residues:
                first_res = residues[0].get_id()[1]
                last_res = residues[-1].get_id()[1]
                chain_info[chain_id] = {
                    'first': first_res,
                    'last': last_res,
                    'length': len(residues)
                }
    
    return chain_info

def analyze_pocket_geometry(pdb_file, ligand_specs, distance_cutoff=5.0):
    """
    Analyze the geometry of the binding pocket formed by multiple ligands
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)
    
    # Get all ligand atoms
    ligand_atoms = []
    ligand_centers = []
    
    for ligand_chain, ligand_resid in ligand_specs:
        lig_atoms_this = []
        for model in structure:
            for chain in model:
                if chain.get_id() == ligand_chain:
                    for residue in chain:
                        if residue.get_id()[1] == ligand_resid:
                            lig_atoms_this = [atom for atom in residue.get_atoms()]
                            break
        
        if lig_atoms_this:
            ligand_atoms.extend(lig_atoms_this)
            coords = np.array([atom.get_coord() for atom in lig_atoms_this])
            center = np.mean(coords, axis=0)
            ligand_centers.append(center)
    
    if len(ligand_centers) < 2:
        return None
    
    # Calculate pocket center (average of all ligand centers)
    pocket_center = np.mean(ligand_centers, axis=0)
    
    # Calculate distance between ligands
    ligand_distances = []
    for i in range(len(ligand_centers)):
        for j in range(i+1, len(ligand_centers)):
            dist = np.linalg.norm(ligand_centers[i] - ligand_centers[j])
            ligand_distances.append(dist)
    
    # Calculate pocket span (max distance from center to any ligand atom)
    max_span = 0
    for atom in ligand_atoms:
        dist = np.linalg.norm(atom.get_coord() - pocket_center)
        max_span = max(max_span, dist)
    
    return {
        'pocket_center': pocket_center,
        'ligand_centers': ligand_centers,
        'inter_ligand_distances': ligand_distances,
        'pocket_span': max_span,
        'num_ligands': len(ligand_centers)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare PDB for RFDiffusion')
    parser.add_argument('--input_pdb', required=True, help='Input PDB file')
    parser.add_argument('--output_pdb', required=True, help='Output cleaned PDB file')
    parser.add_argument('--ligands', nargs='+', 
                       help='Ligand specifications as CHAIN:RESID (e.g., C:1 D:2)')
    parser.add_argument('--keep_chains', nargs='+', help='Chains to keep (e.g., A B)')
    parser.add_argument('--distance_cutoff', type=float, default=5.0,
                       help='Distance cutoff for binding residues (Angstroms)')
    
    args = parser.parse_args()
    
    # Get chain information
    print("\n=== Chain Information ===")
    chain_info = get_chain_info(args.input_pdb)
    for chain_id, info in chain_info.items():
        print(f"Chain {chain_id}: residues {info['first']}-{info['last']} (length: {info['length']})")
    
    # Parse ligand specifications
    ligand_specs = []
    if args.ligands:
        print("\n=== Parsing Ligands ===")
        for lig_spec in args.ligands:
            try:
                chain, resid = lig_spec.split(':')
                ligand_specs.append((chain, int(resid)))
                print(f"  Ligand: chain {chain}, residue {resid}")
            except ValueError:
                print(f"  ERROR: Invalid ligand format '{lig_spec}'. Use CHAIN:RESID (e.g., C:1)")
                exit(1)
    
    # Analyze pocket geometry if multiple ligands
    if len(ligand_specs) >= 2:
        print("\n=== Pocket Geometry Analysis ===")
        pocket_info = analyze_pocket_geometry(args.input_pdb, ligand_specs, 
                                              distance_cutoff=args.distance_cutoff)
        if pocket_info:
            print(f"Number of ligands: {pocket_info['num_ligands']}")
            print(f"Pocket center: {pocket_info['pocket_center']}")
            print(f"Pocket span (radius): {pocket_info['pocket_span']:.2f} Å")
            print(f"Inter-ligand distances:")
            for i, dist in enumerate(pocket_info['inter_ligand_distances']):
                print(f"  Ligand pair {i+1}: {dist:.2f} Å")
            
            # Estimate peptide length needed
            max_dist = max(pocket_info['inter_ligand_distances'])
            estimated_peptide_length = int(max_dist / 3.8) + 10  # ~3.8Å per residue in extended
            print(f"\nEstimated peptide length needed: ~{estimated_peptide_length} residues")
            print(f"  (to span {max_dist:.1f} Å between ligands)")
    
    # Find binding pocket residues if ligands specified
    if ligand_specs:
        print("\n=== Binding Pocket Residues ===")
        binding_res = find_ligand_binding_residues(
            args.input_pdb, 
            ligand_specs,
            distance_cutoff=args.distance_cutoff
        )
        print(f"Found {len(binding_res)} residues within {args.distance_cutoff} Å of ligands:")
        
        # Group by chain for better readability
        from collections import defaultdict
        by_chain = defaultdict(list)
        for res in binding_res:
            chain = res[0]
            resnum = res[1:]
            by_chain[chain].append(resnum)
        
        for chain in sorted(by_chain.keys()):
            resnums = ','.join([f"{chain}{r}" for r in by_chain[chain]])
            print(f"  Chain {chain}: {resnums}")
        
        print("\nUse these as hotspot residues in RFDiffusion:")
        print(f"ppi.hotspot_res=[{','.join(binding_res)}]")
    
    # Clean structure
    print("\n=== Cleaning Structure ===")
    remove_ligands(args.input_pdb, args.output_pdb, keep_chains=args.keep_chains)
    
    print("\n=== Next Steps ===")
    print("1. Review the chain information and pocket geometry above")
    print("2. Use the cleaned PDB for RFDiffusion")
    print("3. Set up contig map based on chain lengths")
    if len(ligand_specs) >= 2:
        print(f"4. Consider peptide length: {estimated_peptide_length}±10 residues")
        print(f"   (adjust based on desired pocket coverage)")
