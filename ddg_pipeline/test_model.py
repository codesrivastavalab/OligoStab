"""
test_model.py
Load a trained ddG prediction model and test on mutations

Usage:
    # CSV input: Multiple PDBs with mutations
    python test_model.py --checkpoint checkpoints/my_model.ckpt --csv mutations.csv --pdb-column pdb --mutation-column mutation --output results.csv
    
Or use the Python API directly for more control.
"""

import torch
import argparse
import sys
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import gc

# Add ddg_pipeline directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from datasets import Mutation
from protein_mpnn_utils import parse_PDB
from oligomer_esm_utils import detect_oligomer_type 

torch.cuda.empty_cache()
gc.collect()

# Class labels mapping
CLASS_LABELS = {
    0: -1,  # stable (CE space 0 -> user space -1)
    1: 0,   # neutral (CE space 1 -> user space 0)
    2: 1    # unstable (CE space 2 -> user space 1)
}

CLASS_NAMES = {
    -1: 'stable',
    0: 'neutral',
    1: 'unstable'
}

# 20 canonical amino acids for site-saturation mutagenesis.
AA20 = "ACDEFGHIKLMNPQRSTVWY"


def load_trained_model(checkpoint_path, mode="regression", config_path="configs/config.yaml"):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to .ckpt file
        mode: 'regression' or 'classification'
        config_path: Path to config.yaml
    
    Returns:
        model_pl: Lightning module (contains the compiled model)
        device: torch device
        config: OmegaConf config
    """
    # Import the appropriate model class
    if mode == "classification":
        from train_ddg_model_classification import CompiledModelPL
    else:
        # from train_ddg_model_regression import CompiledModelPL
        from ddg_pipeline.train_ddG_model import CompiledModelPL  
    
    # Load config
    config = OmegaConf.load(config_path)
    try:
        local_config = OmegaConf.load("configs/local.yaml")
        config = OmegaConf.merge(config, local_config)
    except FileNotFoundError:
        print("Warning: local.yaml not found, using only config.yaml")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Mode: {mode}")
    # Allow loading older checkpoints that may miss newer buffers (e.g. ddg_min/max).
    strict_load = False
    model_pl = CompiledModelPL.load_from_checkpoint(
        checkpoint_path,
        config=config,
        strict=strict_load,
    )
    model_pl = model_pl.to(device)
    model_pl.eval()
    
    print("✓ Model loaded and set to eval mode")
    return model_pl, device, config


def parse_mutations_from_strings(mutation_strings):
    """
    Parse mutation strings like 'A10E' into Mutation objects.
    
    Args:
        mutation_strings: List of strings like ['A10E', 'K25R', 'L50F'] 1-indexed format.
        
    Returns:
        List of Mutation objects
    """
    mutations = []
    for mut_str in mutation_strings:
        # Parse format: WildtypePositionMutant (e.g., 'A10E')
        wt_aa = mut_str[0].upper()
        mut_aa = mut_str[-1].upper()
        pos_str = mut_str[1:-1] # in 1-indexing format
        
        try:
            position = int(pos_str)-1  # Convert to 0-indexed
        except ValueError:
            raise ValueError(f"Invalid mutation format: {mut_str}. Expected format like 'A10E'")
        
        mutations.append(Mutation(
            position=position,
            wildtype=wt_aa,
            mutation=mut_aa,
            ddG=None,  # No experimental value for testing
            pdb=''
        ))
    
    return mutations


def load_pdb_structure(pdb_path, config):
    """
    Load a PDB structure file.
    
    Args:
        pdb_path: Path to PDB file
        config: OmegaConf config
        
    Returns:
        pdb_dict: Dictionary with structure info and sequence
    """
    if not Path(pdb_path).exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    
    print(f"Loading PDB from: {pdb_path}")
    pdb = parse_PDB(pdb_path)
    # seq = pdb[0]['seq']
    # print(seq)
    # print('1-based pos 15 ->', seq[15-1])   # compare to your CSV D15K
    
    if 'seq' not in pdb[0]:
        raise ValueError(f"Could not extract sequence from {pdb_path}")
    
    pdb_name = Path(pdb_path).stem
    pdb[0]['name'] = pdb_name
    
    print(f"✓ PDB loaded: {pdb_name}")
    print(f"  Sequence length: {len(pdb[0]['seq'])} residues")
    print(f"  Sequence: {pdb[0]['seq'][:50]}..." if len(pdb[0]['seq']) > 50 else f"  Sequence: {pdb[0]['seq']}")
    
    return pdb


def validate_mutations(mutations, sequence):
    """
    Validate that mutations are within sequence bounds and match wildtype.
    
    Args:
        mutations: List of Mutation objects
        sequence: WT sequence string
        
    Returns:
        List of valid mutations
    """
    valid_mutations = []
    
    for mut in mutations:
        pos = mut.position
        wt_aa = mut.wildtype
        
        # Check bounds
        if pos < 0 or pos >= len(sequence):
            print(f"Skipping {wt_aa}{pos}{mut.mutation}: position {pos} out of bounds (seq length: {len(sequence)})")
            continue
        
        # Check wildtype match
        if sequence[pos] != wt_aa:
            print(f"Warning: {wt_aa}{pos}{mut.mutation} - sequence at position {pos} is '{sequence[pos]}', not '{wt_aa}'")
            # Still include it, but warn the user
        
        valid_mutations.append(mut)
    
    return valid_mutations


def generate_site_saturation_mutations(pdb_record):
    """Generate all single substitutions for monomer, homomer or heteromer.

    Returns:
        mut_list: list[tuple(label, mutant_seq)] of length = full chain length * 19 (excluding WT)
    """
    oligomer_type, chain_ids, chain_sequences = detect_oligomer_type(pdb_record)
    num_chains = len(chain_sequences)

    if oligomer_type == 'monomer':
        full_seq = chain_sequences[0]
        mut_list = [('WT', full_seq)]
        for pos, wt in enumerate(full_seq):
            for aa in AA20:
                if aa == wt:
                    continue
                label = f"{wt}{pos}{aa}"
                mut_seq = full_seq[:pos] + aa + full_seq[pos+1:]
                mut_list.append(('Mono:' + label, mut_seq))
        return mut_list
    
    elif oligomer_type == 'homomer':
        full_seq = chain_sequences[0]
        mut_list = [('WT', full_seq*num_chains)]
        for pos, wt in enumerate(full_seq):
            for aa in AA20:
                if aa == wt:
                    continue
                label = f"{wt}{pos}{aa}"
                mut_seq = full_seq[:pos] + aa + full_seq[pos+1:]
                mut_list.append(('Homo:'+ label, mut_seq*num_chains))  # same mutation in all chains
        return mut_list
    
    else:  # heteromer
        full_seq = ''.join(chain_sequences)
        mut_list = [('WT', full_seq)]
        offset = 0  # keeps track of where each chain starts

        for chain_seq, cid in zip(chain_sequences, chain_ids):
            for i, wt in enumerate(chain_seq):
                global_pos = offset + i

                for aa in AA20:
                    if aa == wt:
                        continue

                    # Use global 0-based position so downstream indexing matches full_seq.
                    label = f"Hetero {cid}:{wt}{global_pos}{aa}"

                    mut_seq = full_seq[:global_pos] + aa + full_seq[global_pos+1:]
                    mut_list.append((label, mut_seq))

            offset += len(chain_seq)
        return mut_list
    

def predict_ddg(model_pl, pdb, mutations=None, mut_list=None, device=None, use_normalization=True):
    """
    Predict ddG for mutations (regression mode).
    
    Supports both normalized and non-normalized checkpoints via the explicit
    use_normalization flag.
    
    Args:
        model_pl: Loaded Lightning module
        pdb: PDB structure dict
        mutations: List of Mutation objects
        mut_list: List of tuples (label, mutant sequence) for all site-saturation mutants
        device: torch device
        
    Returns:
        pred_norm: numpy array of normalized predicted ddG values (shape: [num_mutations])
        pred_denorm: numpy array of denormalized predicted ddG values (shape: [num_mutations])
    """
    oligomer_type, chain_ids, chain_sequences = detect_oligomer_type(pdb[0])
    with torch.no_grad():
        # Call the model (model_pl.forward calls model_pl.model which is StabilityPredictionModel)
        fwd_pred_ddg, rev_pred_ddg = model_pl(pdb, mutations=mutations, mut_list=mut_list, pdb_type=oligomer_type, num_chains=len(chain_sequences))
        pred = fwd_pred_ddg.detach()
        
        if use_normalization:
            # Normalized protocol: outputs are in normalized space [-1, 1]
            pred_norm_t = pred
            if hasattr(model_pl, "denormalize_ddg"):
                pred_denorm_t = model_pl.denormalize_ddg(pred_norm_t)
            else:
                pred_denorm_t = pred_norm_t
        else:
            # Non-normalized protocol: outputs are already on the original scale.
            pred_denorm_t = pred
            pred_norm_t = pred  # No normalized version available

    return pred_norm_t.cpu().numpy(), pred_denorm_t.cpu().numpy()


def predict_classes(model_pl, pdb, mutations=None, mut_list=None, device=None):
    """
    Predict stability classes for mutations (classification mode).
    
    Args:
        model_pl: Loaded Lightning module
        pdb: PDB structure dict
        mutations: List of Mutation objects
        device: torch device
        
    Returns:
        predictions: numpy array of predicted class labels in user space {-1, 0, 1} (shape: [num_mutations])
        confidences: numpy array of softmax probabilities (shape: [num_mutations, 3])
    """
    with torch.no_grad():
        # Call the model (returns logits for forward pass)
        fwd_pred_logits, rev_pred_logits = model_pl(pdb, mutations) # shape: [num_mutations, 3]
        # fwd_pred_logits shape: [num_mutations, 3]
        
        # Get class predictions via argmax (in CE space: 0, 1, 2)
        pred_classes_ce = torch.argmax(fwd_pred_logits, dim=-1)  # shape: [num_mutations]
        
        # Convert from CE space {0, 1, 2} to user space {-1, 0, 1}
        pred_classes_user = pred_classes_ce.cpu().numpy() - 1
        
        # Get softmax probabilities for confidence scores
        softmax_probs = torch.softmax(fwd_pred_logits, dim=-1)  # shape: [num_mutations, 3]
        confidences = softmax_probs.cpu().numpy()
    
    return pred_classes_user, confidences


def test_from_csv(checkpoint_path, csv_path, mode="regression", pdb_column='pdb', mutation_column='mutation', config_path=None, pdb_dir=None):
    """
    Test the model on mutations loaded from a CSV file.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        csv_path: Path to CSV file with mutations
        mode: 'regression' or 'classification' (default: 'regression')
        pdb-column: Name of column containing PDB paths/IDs (default: 'pdb')
        mutation-column: Name of column containing mutations (default: 'mutation')
        config_path: Path to config.yaml (optional)
        pdb-dir: Optional directory to prefix to PDB entries in the CSV (if CSV contains basenames/IDs instead of full paths)
    
    Returns:
        DataFrame with results
    
    Example CSV format:
        pdb,mutation,optional_column
        /path/to/1ABC.pdb,A10E,extra_info
        /path/to/1ABC.pdb,K25R,extra_info
        /path/to/2XYZ.pdb,L50F,extra_info
    """
    if config_path is None:
        config_path = "ddg_pipeline/configs/config.yaml"
    
    # Load CSV
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading mutations from: {csv_path}")
    df_input = pd.read_csv(csv_path)
    
    # Validate columns
    if pdb_column not in df_input.columns:
        raise ValueError(f"Column '{pdb_column}' not found in CSV. Available: {list(df_input.columns)}")
    if mutation_column not in df_input.columns:
        raise ValueError(f"Column '{mutation_column}' not found in CSV. Available: {list(df_input.columns)}")
    
    print(f"✓ Loaded {len(df_input)} rows from CSV")
    print(f"  PDB column: '{pdb_column}'")
    print(f"  Mutation column: '{mutation_column}'")
    print(f"  Prediction mode: {mode}")
    
    # Load model
    model_pl, device, config = load_trained_model(checkpoint_path, mode=mode, config_path=config_path)
    use_normalization = bool(config.training.get('use_normalization', True))
    print(f"  Normalization enabled: {use_normalization}")
    
    # Group mutations by PDB (allow prefixing entries with a directory)
    pdb_mutations = {}
    for idx, row in df_input.iterrows():
        raw_entry = str(row[pdb_column])
        # If a pdb_dir was provided, treat CSV entries as basenames/IDs and join
        if pdb_dir is not None:
            candidate = Path(pdb_dir) / raw_entry
            # try candidate as given, or with .pdb suffix if missing
            if candidate.exists():
                pdb_path = str(candidate)
            else:
                candidate_pdb = candidate.with_suffix('.pdb')
                if candidate_pdb.exists():
                    pdb_path = str(candidate_pdb)
                else:
                    # fall back to the joined path (file-not-found will be raised later)
                    pdb_path = str(candidate)
        else:
            pdb_path = raw_entry
        mutation_str = row[mutation_column]
        
        if pdb_path not in pdb_mutations:
            pdb_mutations[pdb_path] = []
        pdb_mutations[pdb_path].append(mutation_str)
    
    print(f"\n✓ Grouped mutations by {len(pdb_mutations)} unique PDB(s)")
    
    # Process each PDB
    all_results = []
    for pdb_path, mutation_strings in pdb_mutations.items():
        print(f"\n{'='*70}")
        print(f"Processing: {Path(pdb_path).name}")
        print(f"{'='*70}")
        
        try:
            # Load PDB
            pdb_dict = load_pdb_structure(pdb_path, config)
            
            # Debug: print oligomer type detection
            oligomer_type, chain_ids, chain_sequences = detect_oligomer_type(pdb_dict[0])
            print(f"  Detected oligomer type: {oligomer_type} (chains: {chain_ids}, lengths: {[len(s) for s in chain_sequences]})")
            
            # Parse and validate mutations
            mutations = parse_mutations_from_strings(mutation_strings)
            print(f"\nParsed {len(mutations)} mutations for this PDB:")
            for mut in mutations:
                print(f"  {mut.wildtype}{mut.position}{mut.mutation}")
            
            valid_mutations = validate_mutations(mutations, pdb_dict[0]['seq'])
            if len(valid_mutations) != len(mutations):
                print(f"\n⚠ Using {len(valid_mutations)}/{len(mutations)} valid mutations")
            
            if not valid_mutations:
                print(f"⚠ Skipping {Path(pdb_path).name} - no valid mutations")
                continue
            
            # Predict
            print(f"\n⏳ Running predictions ({len(valid_mutations)} mutations)...")
            if mode == "classification":
                predictions, confidences = predict_classes(model_pl, pdb_dict, valid_mutations, device)
                
                # Format results for classification
                for mut, pred_class, conf_scores in zip(valid_mutations, predictions, confidences):
                    all_results.append({
                        'pdb': pdb_dict[0]['name'],
                        'pdb_path': str(pdb_path),
                        'position': mut.position,
                        'wildtype': mut.wildtype,
                        'mutation': mut.mutation,
                        'mutation_str': f"{mut.wildtype}{mut.position}{mut.mutation}",
                        'predicted_class': int(pred_class),
                        'predicted_class_name': CLASS_NAMES[int(pred_class)],
                        'conf_stable': float(conf_scores[0]),
                        'conf_neutral': float(conf_scores[1]),
                        'conf_unstable': float(conf_scores[2]),
                    })
            else:
                pred_norm, pred_denorm = predict_ddg(
                    model_pl,
                    pdb_dict,
                    mutations=valid_mutations,
                    device=device,
                    use_normalization=use_normalization,
                )
                
                # Format results for regression
                for mut, pred_ddg_norm, pred_ddg_denorm in zip(valid_mutations, pred_norm, pred_denorm):
                    all_results.append({
                        'pdb': pdb_dict[0]['name'],
                        'pdb_path': str(pdb_path),
                        'position': mut.position,
                        'wildtype': mut.wildtype,
                        'mutation': mut.mutation,
                        'mutation_str': f"{mut.wildtype}{mut.position}{mut.mutation}",
                        'predicted_ddG_normalized': float(pred_ddg_norm),
                        'predicted_ddG_denormalized': float(pred_ddg_denorm),
                    })
            
            print(f"✓ Completed {Path(pdb_path).name}")
        
        except Exception as e:
            print(f"❌ Error processing {Path(pdb_path).name}: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        raise ValueError("No results generated - check input data and PDB files")
    
    df_results = pd.DataFrame(all_results)
    
    print("\n" + "="*70)
    print("FINAL PREDICTION RESULTS")
    print("="*70)
    print(df_results.to_string(index=False))
    print("="*70)
    
    return df_results


def test_from_csv_single_pdb(checkpoint_path, pdb_path, csv_path, mode="regression", mutation_column='mutation', config_path=None):
    """
    Test the model on mutations loaded from a CSV file, all from a single PDB.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        pdb_path: Path to PDB structure file
        csv_path: Path to CSV file with mutations (only needs mutation column)
        mode: 'regression' or 'classification' (default: 'regression')
        mutation_column: Name of column containing mutations (default: 'mutation')
        config_path: Path to config.yaml (optional)
    
    Returns:
        DataFrame with results
    
    Example CSV format (single column):
        mutation
        A10E
        K25R
        L50F
    """
    if config_path is None:
        config_path = "configs/config.yaml"
    
    # Load CSV
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading mutations from: {csv_path}")
    df_input = pd.read_csv(csv_path)
    
    # Validate mutation column
    if mutation_column not in df_input.columns:
        raise ValueError(f"Column '{mutation_column}' not found in CSV. Available: {list(df_input.columns)}")
    
    print(f"✓ Loaded {len(df_input)} rows from CSV")
    print(f"  Mutation column: '{mutation_column}'")
    print(f"  PDB file: {pdb_path}")
    print(f"  Prediction mode: {mode}")
    
    # Load model
    model_pl, device, config = load_trained_model(checkpoint_path, mode=mode, config_path=config_path)
    use_normalization = bool(config.training.get('use_normalization', True))
    print(f"  Normalization enabled: {use_normalization}")
    
    # Extract mutation strings from CSV
    mutation_strings = df_input[mutation_column].tolist()
    
    print(f"\n{'='*70}")
    print(f"Processing: {Path(pdb_path).name}")
    print(f"{'='*70}")
    
    # Load PDB
    pdb = load_pdb_structure(pdb_path, config)
    
    # Parse and validate mutations
    mutations = parse_mutations_from_strings(mutation_strings)
    print(f"\nParsed {len(mutations)} mutations:")
    for mut in mutations:
        print(f"  {mut.wildtype}{mut.position}{mut.mutation}")
    
    valid_mutations = validate_mutations(mutations, pdb[0]['seq'])
    if len(valid_mutations) != len(mutations):
        print(f"\n Using {len(valid_mutations)}/{len(mutations)} valid mutations")
    
    if not valid_mutations:
        raise ValueError("No valid mutations found - check your CSV and PDB file")
    
    # Predict
    print(f"\n Running predictions ({len(valid_mutations)} mutations)...")
    if mode == "classification":
        predictions, confidences = predict_classes(model_pl, pdb, valid_mutations, device)
        
        # Format results for classification
        all_results = []
        for mut, pred_class, conf_scores in zip(valid_mutations, predictions, confidences):
            all_results.append({
                'pdb': pdb[0]['name'],
                'position': mut.position,
                'wildtype': mut.wildtype,
                'mutation': mut.mutation,
                'mutation_str': f"{mut.wildtype}{mut.position}{mut.mutation}",
                'predicted_class': int(pred_class),
                'predicted_class_name': CLASS_NAMES[int(pred_class)],
                'conf_stable': f"{conf_scores[0]:.3f}",
                'conf_neutral': f"{conf_scores[1]:.3f}",
                'conf_unstable': f"{conf_scores[2]:.3f}",
            })
    else:
        pred_norm, pred_denorm = predict_ddg(
            model_pl,
            pdb,
            mutations=valid_mutations,
            device=device,
            use_normalization=use_normalization,
        )
        
        # Format results for regression
        all_results = []
        for mut, pred_ddg_norm, pred_ddg_denorm in zip(valid_mutations, pred_norm, pred_denorm):
            all_results.append({
                'pdb': pdb[0]['name'],
                # 'pdb_path': str(pdb_path),
                'position': mut.position,
                'wildtype': mut.wildtype,
                'mutation': mut.mutation,
                'mutation_str': f"{mut.wildtype}{mut.position}{mut.mutation}",
                'predicted_ddG_normalized': float(pred_ddg_norm),
                'predicted_ddG_denormalized': float(pred_ddg_denorm),
            })
    
    df_results = pd.DataFrame(all_results)
    
    print("\n" + "="*70)
    print("FINAL PREDICTION RESULTS")
    print("="*70)
    print(df_results.to_string(index=False))
    print("="*70)
    
    return df_results


def test_all_mutations_single_pdb(checkpoint_path, pdb_path, mode="regression", config_path=None):
    """Run inference on all single substitutions for a single (possibly oligomeric) target."""
    if config_path is None:
        config_path = "configs/config.yaml"

    print(f"PDB file: {pdb_path}")
    print(f"Prediction mode: {mode}")

    model_pl, device, config = load_trained_model(checkpoint_path, mode=mode, config_path=config_path)

    print(f"\n{'='*70}")
    print(f"Processing: {Path(pdb_path).name}")
    print(f"{'='*70}")

    pdb = load_pdb_structure(pdb_path, config)

    oligomer_type, chain_ids, chain_sequences = detect_oligomer_type(pdb[0])
    print(f"Detected target type: {oligomer_type} (chains: {', '.join(chain_ids)})")
    use_normalization = bool(config.training.get('use_normalization', True))
    print(f"Normalization enabled: {use_normalization}")

    # Auto-enable oligomer ESM path for multi-chain targets when available.
    num_chains = int(pdb[0].get('num_of_chains', 1))
    if num_chains > 1 and hasattr(model_pl, 'config'):
        if hasattr(model_pl.config, 'model') and hasattr(model_pl.config.model, 'use_oligomer_esm'):
            model_pl.config.model.use_oligomer_esm = True
            print("Enabled model.use_oligomer_esm=True for multi-chain target")

    mut_list = generate_site_saturation_mutations(pdb[0])
    print(f"Generated {len(mut_list)} single mutants across {num_chains} chain(s)")

    if not mut_list:
        raise ValueError("No mutations generated from PDB sequence")

    print(f"\nRunning predictions ({len(mut_list)} mutations)...")

    pred_norm, pred_denorm = predict_ddg(
        model_pl,
        pdb,
        mut_list=mut_list,
        device=device,
        use_normalization=use_normalization,
    )
    mutant_entries = [(label, seq) for (label, seq) in mut_list if label != 'WT']

    if len(pred_norm) != len(mutant_entries):
        raise ValueError(
            f"Prediction count ({len(pred_norm)}) does not match mutant count ({len(mutant_entries)})."
        )

    all_results = []
    for (label, _mut_seq), pred_ddg_norm, pred_ddg_denorm in zip(mutant_entries, pred_norm, pred_denorm):
        chid, mut_str = label.split(':', 1)
        all_results.append({
            'pdb': pdb[0]['name'],
            'chain_id': chid,
            'position': mut_str[1:-1],  
            'wildtype': mut_str[0],
            'mutation': mut_str[-1],
            'mutation_str': mut_str,
            'predicted_ddG_normalized': float(pred_ddg_norm),
            'predicted_ddG_denormalized': float(pred_ddg_denorm),
        })

    df_results = pd.DataFrame(all_results)
    print("\n" + "="*70)
    print("FINAL PREDICTION RESULTS")
    print("="*70)
    print(df_results.to_string(index=False))
    print("="*70)
    return df_results


def save_results(df_results, output_path):
    """Save results to CSV."""
    df_results.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test trained ddG prediction model on mutations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Mode 1: Single PDB with command-line mutations
  python test_model.py \\
    --checkpoint checkpoints/my_model.ckpt \\
    --pdb pdbs/1ABC.pdb \\
    --mutations A10E K25R L50F

  # Mode 2: Single PDB with mutations from CSV (simpler for many mutations)
  python test_model.py \\
    --checkpoint checkpoints/my_model.ckpt \\
    --pdb pdbs/1ABC.pdb \\
    --csv mutations.csv \\
    --output results.csv

  # Mode 3: Multiple PDBs with mutations from CSV
  python test_model.py \\
    --checkpoint checkpoints/my_model.ckpt \\
    --csv mutations.csv \\
    --pdb-column pdb_file \\
    --mutation-column mutation \\
    --output results.csv

CSV formats:
  Mode 2 (single PDB):
    mutation
    A10E
    K25R
    L50F
  
  Mode 3 (multiple PDBs):
    pdb_file,mutation,other_info
    /path/to/1ABC.pdb,A10E,info1
    /path/to/1ABC.pdb,K25R,info2
    /path/to/2XYZ.pdb,L50F,info3
        """
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "--pdb",
        default=None,
        help="Path to PDB structure file"
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to CSV file with mutations"
    )
    parser.add_argument(
        "--pdb-column",
        default="pdb",
        help="Name of CSV column containing PDB paths (default: 'pdb', used in Mode 3)"
    )
    parser.add_argument(
        "--pdb-dir",
        default=None,
        help="Optional directory to prepend to PDB IDs/filenames listed in the CSV"
    )
    parser.add_argument(
        "--mutation-column",
        default="mutation",
        help="Name of CSV column containing mutations (default: 'mutation')"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--mode",
        choices=["regression", "classification"],
        default="regression",
        help="Prediction mode: 'regression' for ddG values or 'classification' for stability classes (default: regression)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save results CSV (optional)"
    )
    parser.add_argument(
        "--all-mutations",
        action="store_true",
        help="Generate and predict all single substitutions across all chains (requires --pdb)"
    )
    
    args = parser.parse_args()
    
    if args.all_mutations:
        if args.pdb is None:
            raise ValueError("--all-mutations requires --pdb")
        print("Mode 0: Single PDB, all site-saturation mutations")
        df_results = test_all_mutations_single_pdb(
            args.checkpoint,
            args.pdb,
            mode=args.mode,
            config_path=args.config,
        )
    # CSV mode - could be Mode 1 or Mode 2
    elif args.pdb is not None:
        # Mode 1: Single PDB with CSV mutations
        print("Mode 1: Single PDB with CSV mutations")
        df_results = test_from_csv_single_pdb(
            args.checkpoint,
            args.pdb,
            args.csv,
            mode=args.mode,
            mutation_column=args.mutation_column,
            config_path=args.config
        )
    else:
        # Mode 2: Multiple PDBs from CSV
        print("Mode 2: Multiple PDBs with CSV mutations")
        df_results = test_from_csv(
            args.checkpoint,
            args.csv,
            mode=args.mode,
            pdb_column=args.pdb_column,
            mutation_column=args.mutation_column,
            config_path=args.config,
            pdb_dir=args.pdb_dir,
        )
    
    # Save if requested
    if args.output:
        save_results(df_results, args.output)


if __name__ == "__main__":
    main()
