import torch

def detect_oligomer_type(pdb_record):
    """Classify target as monomer, homomer, or heteromer from seq_chain_* fields.

    Returns:
        oligomer_type: one of {'monomer', 'homomer', 'heteromer'}
        chain_ids: ordered chain IDs used for classification
        chain_sequences: ordered chain sequences
    """
    all_seqs, all_keys = get_ordered_chain_sequences(pdb_record)
    if len(all_seqs) <= 1:
        return 'monomer', ['A'], all_seqs
    elif len(set(all_seqs)) == 1:
        return 'homomer', [key.rsplit('_',1)[-1] for key in all_keys], all_seqs
    return 'heteromer', [key.rsplit('_',1)[-1] for key in all_keys], all_seqs


def get_ordered_chain_sequences(pdb_record):
    """Extract chain sequences in deterministic order from a parsed PDB record.

    Args:
        pdb_record: Single dict entry like pdb[0] from parse_PDB output.

    Returns:
        chain_sequences: list[str] in sorted seq_chain_ key order
        chain_keys: list[str] corresponding keys
    """
    chain_keys = sorted([k for k in pdb_record.keys() if k.startswith("seq_chain_")])
    if len(chain_keys) > 0:
        return [pdb_record[k] for k in chain_keys], chain_keys

    # Fallback for monomers or records without explicit per-chain keys.
    return [pdb_record["seq"]], []


def prepare_oligomer_inputs(chain_sequences, tokenizer, gap_size):
    """Create ESM token ids and position ids with inter-chain positional jumps.

    Args:
        chain_sequences: list[str], one sequence per chain
        tokenizer: HuggingFace EsmTokenizer for tokenization
        gap_size: position-id jump inserted between chains

    Returns:
        tokens: LongTensor shape [1, total_tokens_with_specials]
        position_ids: LongTensor shape [1, total_tokens_with_specials]
    """
    all_residue_tokens = []
    all_position_ids = []
    current_position = 0

    for seq in chain_sequences:
        # Use HuggingFace tokenizer for chain-local tokens
        encoded = tokenizer.encode(seq, return_tensors="pt")
        chain_token_ids = encoded[0].tolist()[1:-1] # Remove CLS (first token) and EOS (last token) from the encoded sequence
        chain_length = len(chain_token_ids)

        positions = list(range(current_position, current_position + chain_length))
        all_residue_tokens.extend(chain_token_ids)
        all_position_ids.extend(positions)
        current_position += chain_length + gap_size

    # Build final token sequence with CLS at start and EOS at end
    final_tokens = [tokenizer.cls_token_id] + all_residue_tokens + [tokenizer.eos_token_id]

    # Shift positions to account for CLS token at index 0
    shifted_positions = [p + 1 for p in all_position_ids]
    eos_position = shifted_positions[-1] + 1 if shifted_positions else 1
    final_position_ids = [0] + shifted_positions + [eos_position]

    final_tokens = torch.tensor([final_tokens], dtype=torch.long)
    final_position_ids = torch.tensor([final_position_ids], dtype=torch.long)
    return final_tokens, final_position_ids


def build_oligomer_tokens(pdb_record, tokenizer, gap_size, full_seq=None):
    """Build oligomer-aware ESM tokens.

    Args:
        pdb_record: parsed record with seq_chain_ and seq fields
        tokenizer: HuggingFace EsmTokenizer for tokenization
        gap_size: positional jump between chains
        full_seq: optional concatenated sequence for a specific mutant. When
            provided, it is split by chain lengths (in deterministic chain
            order) so mutant residues are preserved in tokenization.
    """
    chain_sequences, _ = get_ordered_chain_sequences(pdb_record)

    if full_seq is not None:
        expected_len = sum(len(seq) for seq in chain_sequences)
        if len(full_seq) != expected_len:
            raise ValueError(
                f"full_seq length ({len(full_seq)}) does not match expected "
                f"concatenated chain length ({expected_len})"
            )

        # Split concatenated full_seq by WT chain lengths to preserve
        # deterministic chain ordering and include mutant residues.
        split_sequences = []
        start = 0
        for wt_chain_seq in chain_sequences:
            end = start + len(wt_chain_seq)
            split_sequences.append(full_seq[start:end])
            start = end
        chain_sequences = split_sequences

    return prepare_oligomer_inputs(chain_sequences, tokenizer, gap_size=gap_size)
