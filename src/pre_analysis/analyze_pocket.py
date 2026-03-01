#!/usr/bin/env python3

"""
Analyze homotetramer structure and binding pockets
Helps determine:
1. Structure symmetry (D2, C4, etc.)
2. Number of equivalent binding sites
3. Which chains form each binding pocket
4. Recommended strategy for peptide design

Created by: Yonglan Liu
Date: 2026-02-06
"""

from Bio import PDB
import numpy as np
import argparse
from itertools import combinations
from collections import defaultdict

def get_chain_com(structure, chain_id):
    """Calculate center of mass for a chain"""
    coords = []
    masses = []
    
    for model in structure:
        for chain in model:
            if chain.get_id() == chain_id:
                for residue in chain:
                    if residue.get_id()[0] == ' ':
                        for atom in residue:
                            coords.append(atom.get_coord())
                            mass = {'C': 12, 'N': 14, 'O': 16, 'S': 32}.get(atom.element, 12)
                            masses.append(mass)
    
    if not coords:
        return None
    
    coords = np.array(coords)
    masses = np.array(masses)
    com = np.average(coords, axis=0, weights=masses)
    
    return com

def analyze_structure_symmetry(pdb_file, chain_ids):
    """Analyze the symmetry of the structure"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    # Get center of mass for each chain
    chain_coms = {}
    for chain_id in chain_ids:
        com = get_chain_com(structure, chain_id)
        if com is not None:
            chain_coms[chain_id] = com
    
    if len(chain_coms) != len(chain_ids):
        print(f"Warning: Expected 4 chains, found {len(chain_coms)}")
        return None
    
    # Calculate tetramer center
    center = np.mean(list(chain_coms.values()), axis=0)
    
    # Calculate distances from center
    distances = {}
    for chain_id, com in chain_coms.items():
        distances[chain_id] = np.linalg.norm(com - center)
    
    # Analyze arrangement
    results = {
        'center of mass': center,
        'chain_centers': chain_coms,
        'distances_from_center': distances,
        'pairwise_distances': {}
    }
    
    # Calculate all pairwise distances
    for c1, c2 in combinations(chain_ids, 2):
        if c1 in chain_coms and c2 in chain_coms:
            dist = np.linalg.norm(chain_coms[c1] - chain_coms[c2])
            results['pairwise_distances'][f"{c1}-{c2}"] = dist
    
    return results

def identify_binding_interfaces(pdb_file, chain_ids, ligand_specs, distance_cutoff=8.0):
    """
    Identify which chain interfaces are near ligands
    This helps determine of monomer/dimer/trimer/multimer interfaces form the binding pocket
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)
    
    # Get ligand centers
    ligand_centers = []
    for lig_chain, lig_resid in ligand_specs:
        for model in structure:
            for chain in model:
                if chain.get_id() == lig_chain:
                    for residue in chain:
                        if residue.get_id()[1] == lig_resid:
                            coords = np.array([a.get_coord() for a in residue.get_atoms()])
                            ligand_centers.append({
                                'chain': lig_chain,
                                'resid': lig_resid,
                                'center': np.mean(coords, axis=0)
                            })
    
    if not ligand_centers:
        return None
    
    # For each ligand, find which protein chains are nearby
    ligand_chain_contacts = []
    
    for lig_info in ligand_centers:
        lig_center = lig_info['center']
        nearby_chains = []
        
        for chain_id in chain_ids:
            for model in structure:
                for chain in model:
                    if chain.get_id() == chain_id:
                        for residue in chain:
                            if residue.get_id()[0] == ' ':
                                for atom in residue:
                                    dist = np.linalg.norm(atom.get_coord() - lig_center)
                                    if dist < distance_cutoff:
                                        if chain_id not in nearby_chains:
                                            nearby_chains.append(chain_id)
                                        break
        
        ligand_chain_contacts.append({
            'ligand': f"{lig_info['chain']}:{lig_info['resid']}",
            'center': lig_center,
            'contacting_chains': nearby_chains,
            'interface_type': len(nearby_chains)
        })
    
    return ligand_chain_contacts

def recommend_design_strategy(symmetry_info, interface_info):
    """
    Based on symmetry and binding site analysis, recommend design strategy
    """
    recommendations = []
    
    # Analyze how many unique binding sites exist
    if interface_info:
        interface_types = [x['interface_type'] for x in interface_info]
        unique_interfaces = set([tuple(sorted(x['contacting_chains'])) for x in interface_info])
        
        recommendations.append(f"Number of ligands: {len(interface_info)}")
        recommendations.append(f"Number of unique binding interfaces: {len(unique_interfaces)}")
        
        for i, iface_info in enumerate(interface_info):
            chains_str = '+'.join(iface_info['contacting_chains'])
            recommendations.append(f"\nLigand {i+1} ({iface_info['ligand']}):")
            recommendations.append(f"  Contacts chains: {chains_str}")
            recommendations.append(f"  Interface type: {iface_info['interface_type']}-chain interface")
        
        # Design strategy
        recommendations.append("\n" + "="*80)
        recommendations.append("RECOMMENDED DESIGN STRATEGY:")
        recommendations.append("="*80)
        
        if len(unique_interfaces) == 1:
            # All ligands in equivalent pockets
            recommendations.append("✓ All ligands bind to equivalent pockets (symmetric)")
            recommendations.append("✓ Design ONE peptide that will bind to ONE pocket")
            recommendations.append("✓ Due to symmetry, it may bind to multiple equivalent sites")
        else:
            # Different binding sites
            recommendations.append("⚠ Ligands bind to non-equivalent pockets")
            recommendations.append("⚠ You may need to design peptides for each unique pocket")
            recommendations.append("⚠ Or choose one pocket as your primary target")
        
        # Determine which chains to include in RFDiffusion
        recommendations.append("\nRFDIFFUSION SETUP:")
        
        # Get the most common interface
        most_common_interface = max(unique_interfaces, key=lambda x: sum(
            1 for info in interface_info if tuple(sorted(info['contacting_chains'])) == x
        ))
        
        recommendations.append(f"Target interface involves chains: {'+'.join(most_common_interface)}")
        
        if len(most_common_interface) == 2:
            recommendations.append("\n✓ DIMER INTERFACE - Use these chains in contig map:")
            recommendations.append(f"  '{most_common_interface[0]}1-N/{most_common_interface[1]}1-N/peptide_length'")
            recommendations.append("\nNote: This is a dimer interface within the tetramer")
            
        elif len(most_common_interface) == 3:
            recommendations.append("\n✓ TRIMER INTERFACE - Use these chains in contig map:")
            recommendations.append(f"  '{most_common_interface[0]}1-N/{most_common_interface[1]}1-N/{most_common_interface[2]}1-N/peptide_length'")
            
        elif len(most_common_interface) == 4:
            recommendations.append("\n✓ FULL TETRAMER INTERFACE - Use all chains in contig map:")
            recommendations.append(f"  'A1-N/B1-N/C1-N/D1-N/peptide_length'")
            recommendations.append("\nWarning: This is a large complex. Consider:")
            recommendations.append("  - Using only a subset of chains if possible")
            recommendations.append("  - Longer peptides (40-60 residues)")
            recommendations.append("  - Multiple rounds of design")
    
    return recommendations

def print_tetramer_analysis(pdb_file, chain_ids, ligand_specs):
    """Complete analysis of homotetramer"""
    
    print("="*80)
    print("BINDING POCKET ANALYSIS")
    print("="*80)
    print()
    
    # Symmetry analysis
    print("ANALYZING STRUCTURE SYMMETRY...")
    symmetry_info = analyze_structure_symmetry(pdb_file, chain_ids)
    
    if symmetry_info:
        print("\nChain arrangement (COM-COM distance):")
        print("-"*80)
        for chain_id in sorted(chain_ids):
            if chain_id in symmetry_info['distances_from_center']:
                dist = symmetry_info['distances_from_center'][chain_id]
                print(f"Chain {chain_id}: {dist:.2f} Å from the whole structure")
        
        print("\nPairwise chain distances (COM-COM distance):")
        print("-"*80)
        for pair, dist in sorted(symmetry_info['pairwise_distances'].items()):
            print(f"{pair}: {dist:.2f} Å")
    
    # Interface analysis
    print("\n" + "="*80)
    print("ANALYZING BINDING INTERFACES...")
    print("="*80)
    
    interface_info = identify_binding_interfaces(pdb_file, chain_ids, ligand_specs)
    
    if interface_info:
        for info in interface_info:
            print(f"\nLigand {info['ligand']}:")
            print(f"  Contacting chains: {', '.join(info['contacting_chains'])}")
            print(f"  Interface type: {info['interface_type']}-chain interface")
    
    # Recommendations
    print("\n" + "="*80)
    recommendations = recommend_design_strategy(symmetry_info, interface_info)
    for rec in recommendations:
        print(rec)
    
    return symmetry_info, interface_info, recommendations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze homotetramer structure and binding pockets'
    )
    parser.add_argument('--pdb', required=True, help='Input PDB file')
    parser.add_argument('--chains', nargs='+', required=True,
                       help='Tetramer chain IDs (e.g., A B C D)')
    parser.add_argument('--ligands', nargs='+', required=True,
                       help='Ligand specifications as CHAIN:RESID (e.g., E:1 F:1)')
    parser.add_argument('--output', default='tetramer_analysis.txt',
                       help='Output report file')
    
    args = parser.parse_args()
    
    # Parse ligand specifications
    ligand_specs = []
    for lig_spec in args.ligands:
        try:
            chain, resid = lig_spec.split(':')
            ligand_specs.append((chain, int(resid)))
        except ValueError:
            print(f"ERROR: Invalid ligand format '{lig_spec}'. Use CHAIN:RESID")
            exit(1)
    
    if len(args.chains) != 4:
        print(f"Warning: Expected 4 chains for homotetramer, got {len(args.chains)}")
    
    # Run analysis
    print_tetramer_analysis(args.pdb, args.chains, ligand_specs)
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review the interface analysis above")
    print("2. Decide which pocket/interface to target")
    print("3. Update RFDiffusion contig map based on recommendations")
    print("4. Run preparing structure:")
    print(f"  python scripts/prepare_structure.py --input_pdb {args.pdb} --output_pdb [path/to/ouput_pdb] --ligands {' '.join(args.ligands)} --keep_chains {' '.join(args.chains)} --distance_cutoff 4")
