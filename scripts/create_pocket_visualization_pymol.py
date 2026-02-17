#!/usr/bin/env python3
"""
Detailed analysis of ligand binding pocket
Creates visualization scripts and detailed geometry reports

Created by: Yonglan Liu
Date: 2026-02-06
"""

from Bio import PDB
import numpy as np
import argparse
from collections import defaultdict

def analyze_detailed_pocket(pdb_file, ligand_specs, distance_cutoff=5.0):
    """
    Comprehensive analysis of binding pocket with ligands
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)
    
    results = {
        'ligands': [],
        'binding_residues': defaultdict(lambda: {'chains': set(), 'distances': []}),
        'pocket_properties': {}
    }
    
    # Analyze each ligand
    all_ligand_atoms = []
    
    for lig_idx, (ligand_chain, ligand_resid) in enumerate(ligand_specs):
        lig_data = {
            'chain': ligand_chain,
            'resid': ligand_resid,
            'atoms': [],
            'center': None,
            'name': None
        }
        
        for model in structure:
            for chain in model:
                if chain.get_id() == ligand_chain:
                    for residue in chain:
                        if residue.get_id()[1] == ligand_resid:
                            lig_data['name'] = residue.get_resname()
                            lig_data['atoms'] = [atom for atom in residue.get_atoms()]
                            coords = np.array([a.get_coord() for a in lig_data['atoms']])
                            lig_data['center'] = np.mean(coords, axis=0)
                            all_ligand_atoms.extend(lig_data['atoms'])
                            break
        
        results['ligands'].append(lig_data)
    
    # Calculate inter-ligand distances
    inter_ligand_dist = []
    for i in range(len(results['ligands'])):
        for j in range(i+1, len(results['ligands'])):
            if results['ligands'][i]['center'] is not None and results['ligands'][j]['center'] is not None:
                dist = np.linalg.norm(
                    results['ligands'][i]['center'] - results['ligands'][j]['center']
                )
                inter_ligand_dist.append({
                    'ligand1': i,
                    'ligand2': j,
                    'distance': dist
                })
    
    # Find binding residues and their contributions
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            for residue in chain:
                if residue.get_id()[0] == ' ':  # Protein residue
                    res_id = f"{chain_id}{residue.get_id()[1]}"
                    min_dist = float('inf')
                    
                    for atom in residue.get_atoms():
                        for lig_atom in all_ligand_atoms:
                            dist = atom - lig_atom
                            if dist < distance_cutoff:
                                results['binding_residues'][res_id]['chains'].add(chain_id)
                                min_dist = min(min_dist, dist)
                    
                    if min_dist < float('inf'):
                        results['binding_residues'][res_id]['distances'].append(min_dist)
                        results['binding_residues'][res_id]['resname'] = residue.get_resname()
    
    # Calculate pocket center and dimensions
    if all_ligand_atoms:
        all_lig_coords = np.array([a.get_coord() for a in all_ligand_atoms])
        pocket_center = np.mean(all_lig_coords, axis=0)
        
        # Calculate pocket dimensions
        max_x = np.max(all_lig_coords[:, 0]) - np.min(all_lig_coords[:, 0])
        max_y = np.max(all_lig_coords[:, 1]) - np.min(all_lig_coords[:, 1])
        max_z = np.max(all_lig_coords[:, 2]) - np.min(all_lig_coords[:, 2])
        
        results['pocket_properties'] = {
            'center': pocket_center,
            'dimensions': (max_x, max_y, max_z),
            'volume_estimate': max_x * max_y * max_z,
            'max_span': max(max_x, max_y, max_z),
            'inter_ligand_distances': inter_ligand_dist
        }
    
    return results

def print_detailed_report(results, output_file=None):
    """
    Print detailed pocket analysis report
    """
    lines = []
    
    lines.append("="*80)
    lines.append("MULTI-LIGAND BINDING POCKET ANALYSIS")
    lines.append("="*80)
    lines.append("")
    
    # Ligand information
    lines.append("LIGANDS IN POCKET:")
    lines.append("-"*80)
    for i, lig in enumerate(results['ligands']):
        lines.append(f"Ligand {i+1}:")
        lines.append(f"  Chain: {lig['chain']}, Residue: {lig['resid']}")
        lines.append(f"  Name: {lig['name']}")
        lines.append(f"  Number of atoms: {len(lig['atoms'])}")
        if lig['center'] is not None:
            lines.append(f"  Center: ({lig['center'][0]:.2f}, {lig['center'][1]:.2f}, {lig['center'][2]:.2f})")
        lines.append("")
    
    # Inter-ligand distances
    if results['pocket_properties'].get('inter_ligand_distances'):
        lines.append("INTER-LIGAND DISTANCES:")
        lines.append("-"*80)
        for dist_info in results['pocket_properties']['inter_ligand_distances']:
            l1 = dist_info['ligand1'] + 1
            l2 = dist_info['ligand2'] + 1
            lines.append(f"Ligand {l1} ↔ Ligand {l2}: {dist_info['distance']:.2f} Å")
        lines.append("")
    
    # Pocket properties
    if results['pocket_properties']:
        props = results['pocket_properties']
        lines.append("POCKET GEOMETRY:")
        lines.append("-"*80)
        lines.append(f"Pocket center: ({props['center'][0]:.2f}, {props['center'][1]:.2f}, {props['center'][2]:.2f})")
        lines.append(f"Dimensions (Å): X={props['dimensions'][0]:.2f}, Y={props['dimensions'][1]:.2f}, Z={props['dimensions'][2]:.2f}")
        lines.append(f"Maximum span: {props['max_span']:.2f} Å")
        lines.append(f"Estimated volume: {props['volume_estimate']:.2f} Å³")
        lines.append("")
        
        # Peptide length estimation
        max_inter_lig = max([d['distance'] for d in props['inter_ligand_distances']])
        min_peptide = int(max_inter_lig / 3.8) + 5
        max_peptide = int(max_inter_lig / 3.5) + 15
        lines.append(f"RECOMMENDED PEPTIDE LENGTH: {min_peptide}-{max_peptide} residues")
        lines.append(f"  (to span {max_inter_lig:.1f} Å between furthest ligands)")
        lines.append(f"  Assumes ~3.5-3.8 Å per residue in extended conformation")
        lines.append("")
    
    # Binding residues
    lines.append("BINDING POCKET RESIDUES:")
    lines.append("-"*80)
    
    # Group by chain
    by_chain = defaultdict(list)
    for res_id, res_data in sorted(results['binding_residues'].items()):
        chain = res_id[0]
        by_chain[chain].append((res_id, res_data))
    
    total_residues = 0
    for chain in sorted(by_chain.keys()):
        residues = by_chain[chain]
        lines.append(f"Chain {chain}: {len(residues)} residues")
        
        # List residues with details
        for res_id, res_data in residues:
            avg_dist = np.mean(res_data['distances'])
            lines.append(f"  {res_id} ({res_data['resname']}): {avg_dist:.2f} Å from ligands")
        
        lines.append("")
        total_residues += len(residues)
    
    lines.append(f"TOTAL BINDING RESIDUES: {total_residues}")
    lines.append("")
    
    # Hotspot recommendation
    lines.append("RECOMMENDED RFDIFFUSION HOTSPOTS:")
    lines.append("-"*80)
    
    # Select top 20% closest residues as hotspots
    all_res_with_dist = [
        (res_id, np.mean(res_data['distances'])) 
        for res_id, res_data in results['binding_residues'].items()
    ]
    all_res_with_dist.sort(key=lambda x: x[1])
    
    num_hotspots = max(5, len(all_res_with_dist) // 5)
    hotspots = [res_id for res_id, _ in all_res_with_dist[:num_hotspots]]
    
    lines.append(f"Top {num_hotspots} residues (closest to ligands):")
    lines.append(','.join(hotspots))
    lines.append("")
    lines.append("Use in RFDiffusion:")
    lines.append(f"ppi.hotspot_res=[{','.join(hotspots)}]")
    lines.append("")
    
    lines.append("="*80)
    
    # Print to console
    report = '\n'.join(lines)
    print(report)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")
    
    return report, hotspots

def create_pymol_script(pdb_file, ligand_specs, binding_residues, output_script="view_pocket.pml"):
    """
    Create PyMOL script to visualize the binding pocket
    """
    script_lines = [
        "# PyMOL script to visualize multi-ligand binding pocket\n",
        f"load {pdb_file}\n",
        "hide everything\n",
        "show cartoon\n",
        "color gray80\n",
        "\n# Show ligands\n"
    ]
    
    for i, (chain, resid) in enumerate(ligand_specs):
        script_lines.append(f"select ligand{i+1}, chain {chain} and resi {resid}\n")
        script_lines.append(f"show sticks, ligand{i+1}\n")
        colors = ['red', 'blue', 'green', 'yellow', 'magenta']
        script_lines.append(f"color {colors[i % len(colors)]}, ligand{i+1}\n")
    
    script_lines.append("\n# Show binding pocket residues\n")
    
    # Group residues by chain
    by_chain = defaultdict(list)
    for res_id in binding_residues:
        chain = res_id[0]
        resnum = res_id[1:]
        by_chain[chain].append(resnum)
    
    for chain, resnums in by_chain.items():
        resi_str = '+'.join(resnums)
        script_lines.append(f"select pocket_chain_{chain}, chain {chain} and resi {resi_str}\n")
        script_lines.append(f"show sticks, pocket_chain_{chain}\n")
        script_lines.append(f"color cyan, pocket_chain_{chain}\n")
    
    script_lines.extend([
        "\n# Select all pocket residues\n",
        f"select pocket, {' or '.join([f'pocket_chain_{c}' for c in by_chain.keys()])}\n",
        "\n# Show surface\n",
        "show surface, pocket\n",
        "set transparency, 0.5\n",
        "\n# Center view\n",
        "zoom ligand*\n",
        "center ligand*\n",
        "\n# Labels\n",
        "label ligand*, resn\n",
        "\nprint 'Pocket visualization loaded!'\n"
    ])
    
    with open(output_script, 'w') as f:
        f.writelines(script_lines)
    
    print(f"PyMOL visualization script saved to: {output_script}")
    print(f"Load in PyMOL with: pymol {output_script}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze multi-ligand binding pocket in detail'
    )
    parser.add_argument('--pdb', required=True, help='Input PDB file')
    parser.add_argument('--ligands', nargs='+', required=True,
                       help='Ligand specifications as CHAIN:RESID (e.g., C:1 C:2)')
    parser.add_argument('--distance_cutoff', type=float, default=3.2,
                       help='Distance cutoff for binding residues (Å)')
    parser.add_argument('--output_report', default='pocket_analysis.txt',
                       help='Output report filename')
    parser.add_argument('--output_pymol', default='view_pocket.pml',
                       help='Output PyMOL script filename')
    
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
    
    print(f"Analyzing pocket with {len(ligand_specs)} ligands...\n")
    
    # Run analysis
    results = analyze_detailed_pocket(args.pdb, ligand_specs, args.distance_cutoff)
    
    # Print report
    report, hotspots = print_detailed_report(results, args.output_report)
    
    # Create PyMOL visualization
    binding_res_ids = list(results['binding_residues'].keys())
    create_pymol_script(args.pdb, ligand_specs, binding_res_ids, args.output_pymol)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Report saved to: {args.output_report}")
    print(f"PyMOL script saved to: {args.output_pymol}")
    print("\nNext steps:")
    print("1. Review the pocket geometry and recommended peptide length")
    print("2. Visualize in PyMOL to confirm pocket structure")
    print(f"3. Use hotspot residues in RFDiffusion design")
