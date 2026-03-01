#!/usr/bin/env python3
"""
Analyze chain lengths in PDB file and provide solutions
Handles common issues:
- Different chain lengths
- Missing residues
- Non-standard numbering
"""

from Bio import PDB
import argparse
from collections import defaultdict

def analyze_chain_details(pdb_file):
    """
    Detailed analysis of each chain in the PDB
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    chain_info = {}
    
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            
            # Get all residues
            residues = [r for r in chain.get_residues() if r.get_id()[0] == ' ']
            
            if not residues:
                continue
            
            # Get residue numbers
            res_numbers = [r.get_id()[1] for r in residues]
            
            first_resnum = min(res_numbers)
            last_resnum = max(res_numbers)
            num_residues = len(residues)
            
            # Check for gaps
            expected_count = last_resnum - first_resnum + 1
            has_gaps = expected_count != num_residues
            
            # Find gaps
            gaps = []
            if has_gaps:
                all_expected = set(range(first_resnum, last_resnum + 1))
                present = set(res_numbers)
                missing = sorted(all_expected - present)
                
                # Group consecutive missing residues
                if missing:
                    gap_start = missing[0]
                    gap_end = missing[0]
                    
                    for i in range(1, len(missing)):
                        if missing[i] == gap_end + 1:
                            gap_end = missing[i]
                        else:
                            if gap_end == gap_start:
                                gaps.append(f"{gap_start}")
                            else:
                                gaps.append(f"{gap_start}-{gap_end}")
                            gap_start = missing[i]
                            gap_end = missing[i]
                    
                    # Add last gap
                    if gap_end == gap_start:
                        gaps.append(f"{gap_start}")
                    else:
                        gaps.append(f"{gap_start}-{gap_end}")
            
            chain_info[chain_id] = {
                'first_resnum': first_resnum,
                'last_resnum': last_resnum,
                'num_residues': num_residues,
                'expected_count': expected_count,
                'has_gaps': has_gaps,
                'gaps': gaps,
                'residue_numbers': res_numbers
            }
    
    return chain_info

def compare_chains(chain_info, expected_chains):
    """
    Compare chains to identify differences
    """
    if not chain_info:
        return None
    
    # Filter to expected chains only
    filtered_info = {k: v for k, v in chain_info.items() if k in expected_chains}
    
    if not filtered_info:
        print(f"ERROR: None of the expected chains {expected_chains} found in PDB!")
        return None
    
    print("="*80)
    print("CHAIN LENGTH ANALYSIS")
    print("="*80)
    print()
    
    # Print details for each chain
    for chain_id in sorted(filtered_info.keys()):
        info = filtered_info[chain_id]
        print(f"Chain {chain_id}:")
        print(f"  Residue range: {info['first_resnum']}-{info['last_resnum']}")
        print(f"  Number of residues present: {info['num_residues']}")
        
        if info['has_gaps']:
            print(f"  ⚠ GAPS DETECTED:")
            print(f"    Expected residues: {info['expected_count']}")
            print(f"    Missing residues: {', '.join(info['gaps'])}")
        else:
            print(f"  ✓ No gaps")
        print()
    
    # Check if all chains have same length
    lengths = [info['num_residues'] for info in filtered_info.values()]
    ranges = [(info['first_resnum'], info['last_resnum']) for info in filtered_info.values()]
    
    all_same_length = len(set(lengths)) == 1
    all_same_range = len(set(ranges)) == 1
    
    print("="*80)
    print("COMPARISON")
    print("="*80)
    print()
    
    if all_same_length and all_same_range:
        print("✓ All chains have identical length and numbering")
        print(f"  Length: {lengths[0]} residues")
        print(f"  Range: {ranges[0][0]}-{ranges[0][1]}")
        return {
            'status': 'identical',
            'length': lengths[0],
            'range': ranges[0]
        }
    
    elif all_same_length:
        print("⚠ Chains have same length but different numbering")
        print(f"  Length: {lengths[0]} residues")
        print()
        print("Chain ranges:")
        for chain_id in sorted(filtered_info.keys()):
            info = filtered_info[chain_id]
            print(f"  Chain {chain_id}: {info['first_resnum']}-{info['last_resnum']}")
        
        return {
            'status': 'same_length_different_numbering',
            'length': lengths[0],
            'chains': filtered_info
        }
    
    else:
        print("✗ Chains have different lengths!")
        print()
        for chain_id in sorted(filtered_info.keys()):
            info = filtered_info[chain_id]
            print(f"  Chain {chain_id}: {info['num_residues']} residues "
                  f"({info['first_resnum']}-{info['last_resnum']})")
        
        return {
            'status': 'different_lengths',
            'chains': filtered_info
        }

def suggest_solutions(comparison_result, chain_info):
    """
    Suggest solutions based on the analysis
    """
    print()
    print("="*80)
    print("RECOMMENDED SOLUTIONS")
    print("="*80)
    print()
    
    if comparison_result['status'] == 'identical':
        # Easy case - chains are identical
        length = comparison_result['length']
        range_start, range_end = comparison_result['range']
        
        print("✓ Your chains are already identical. Use this contig map:")
        print()
        print("# For RFdiffusion with all 4 chains:")
        print(f"CONTIG='A{range_start}-{range_end}/B{range_start}-{range_end}/"
              f"C{range_start}-{range_end}/D{range_start}-{range_end}/"
              f"0 25-40'")
        print()
        print("# Or in your script:")
        print(f"CHAIN_LENGTH={length}")
        print(f"FIRST_RESIDUE={range_start}")
        
    elif comparison_result['status'] == 'same_length_different_numbering':
        # Same length but different numbering - need to renumber
        print("⚠ Chains have same length but different residue numbering.")
        print()
        print("OPTION 1: Renumber PDB (Recommended)")
        print("-" * 80)
        print("Run the renumbering script:")
        print("  python renumber_pdb.py --input your.pdb --output renumbered.pdb")
        print()
        
        print("OPTION 2: Use specific ranges for each chain")
        print("-" * 80)
        chains_info = comparison_result['chains']
        contig_parts = []
        for chain_id in sorted(chains_info.keys()):
            info = chains_info[chain_id]
            contig_parts.append(f"{chain_id}{info['first_resnum']}-{info['last_resnum']}")
        contig_parts.append("0 25-40")
        
        print("CONTIG='" + "/".join(contig_parts) + "'")
        
    else:
        # Different lengths - more complex
        print("✗ Chains have different lengths. This needs investigation.")
        print()
        
        chains_info = comparison_result['chains']
        
        # Find common range
        max_first = max(info['first_resnum'] for info in chains_info.values())
        min_last = min(info['last_resnum'] for info in chains_info.values())
        
        if max_first <= min_last:
            common_length = min_last - max_first + 1
            
            print("OPTION 1: Use common residue range (Recommended)")
            print("-" * 80)
            print(f"All chains have residues {max_first}-{min_last} ({common_length} residues)")
            print()
            print("Contig map using common range:")
            print(f"CONTIG='A{max_first}-{min_last}/B{max_first}-{min_last}/"
                  f"C{max_first}-{min_last}/D{max_first}-{min_last}/0 25-40'")
            print()
            
            print("Or generate trimmed PDB:")
            print(f"  python trim_pdb.py --input your.pdb --output trimmed.pdb "
                  f"--start {max_first} --end {min_last}")
        else:
            print("ERROR: No common residue range found!")
            print("Chains don't overlap enough.")
        
        print()
        print("OPTION 2: Fix/complete the structure")
        print("-" * 80)
        print("Your homotetramer should have identical chains.")
        print("Possible issues:")
        print("  - Missing residues in crystal structure")
        print("  - Truncated chains")
        print("  - PDB processing errors")
        print()
        print("Solutions:")
        print("  1. Check the original PDB source for complete structure")
        print("  2. Use structure modeling to fill missing regions")
        print("  3. Use only the longest complete chain and model others")
        
        print()
        print("OPTION 3: Use subset of chains")
        print("-" * 80)
        
        # Find chains with same length
        length_groups = defaultdict(list)
        for chain_id, info in chains_info.items():
            length_groups[info['num_residues']].append(chain_id)
        
        for length, chains in sorted(length_groups.items(), key=lambda x: -x[0]):
            if len(chains) >= 2:
                print(f"Chains {', '.join(chains)} all have {length} residues")
                print(f"  Use these chains for design:")
                
                # Get range for first chain in group
                first_chain = chains[0]
                info = chains_info[first_chain]
                
                if len(chains) == 2:
                    print(f"  CONTIG='{chains[0]}{info['first_resnum']}-{info['last_resnum']}/"
                          f"{chains[1]}{info['first_resnum']}-{info['last_resnum']}/0 25-40'")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze chain lengths in PDB and suggest solutions'
    )
    parser.add_argument('--pdb', required=True, help='Input PDB file')
    parser.add_argument('--chains', nargs='+', default=['A', 'B', 'C', 'D'],
                       help='Expected chain IDs (default: A B C D)')
    
    args = parser.parse_args()
    
    print(f"Analyzing: {args.pdb}")
    print(f"Expected chains: {', '.join(args.chains)}")
    print()
    
    # Analyze chains
    chain_info = analyze_chain_details(args.pdb)
    
    if not chain_info:
        print("ERROR: No chains found in PDB file!")
        return 1
    
    # Compare chains
    comparison = compare_chains(chain_info, args.chains)
    
    if comparison is None:
        return 1
    
    # Suggest solutions
    suggest_solutions(comparison, chain_info)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
