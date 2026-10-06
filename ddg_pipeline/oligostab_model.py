import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from protein_mpnn_utils import ProteinMPNN, CA_ProteinFeatures, tied_featurize, _S_to_seq, get_mutant_neighbors
from transformers import EsmModel, EsmTokenizer
import os
import numpy as np
import re
from oligomer_esm_utils import build_oligomer_tokens, get_ordered_chain_sequences, detect_oligomer_type


HIDDEN_DIM = 128
EMBED_DIM = 128
VOCAB_DIM = 21
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

MLP = True
SUBTRACT_MUT = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_nested_tensors(obj, target_device):
    """Recursively move tensors in a nested structure to the target device."""
    if torch.is_tensor(obj):
        return obj.to(target_device)
    if isinstance(obj, list):
        return [_move_nested_tensors(item, target_device) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_move_nested_tensors(item, target_device) for item in obj)
    if isinstance(obj, dict):
        return {key: _move_nested_tensors(value, target_device) for key, value in obj.items()}
    return obj


class LoRALinear(nn.Linear):
    """Linear layer with optional low-rank adapters."""

    def __init__(self, in_features, out_features, bias=True, lora_rank=0, lora_alpha=1.0, lora_dropout=0.0):
        super().__init__(in_features, out_features, bias=bias)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = nn.Dropout(lora_dropout)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.lora_rank))
            self.scaling = self.lora_alpha / self.lora_rank
            nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.scaling = 0.0

    def freeze_base_weights(self):
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, input):
        output = F.linear(input, self.weight, self.bias)
        if self.lora_rank > 0:
            lora_input = self.lora_dropout(input)
            lora_update = F.linear(F.linear(lora_input, self.lora_A), self.lora_B)
            output = output + lora_update * self.scaling
        return output

def get_protein_mpnn(config, version='v_48_020.pt'):
    """Loading Pre-trained ProteinMPNN model for structure embeddings"""
    hidden_dim = 128
    num_layers = 3 

    model_weight_dir = os.path.join(config.platform.proteinmpnn_dir, 'vanilla_model_weights')
    checkpoint_path = os.path.join(model_weight_dir, version)
    # checkpoint_path = "vanilla_model_weights/v_48_020.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device) 
    mpnn_model = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, 
                        num_encoder_layers=num_layers, num_decoder_layers=num_layers, k_neighbors=checkpoint['num_edges'], augment_eps=0.0)
    if config.model.load_pretrained:
        mpnn_model.load_state_dict(checkpoint['model_state_dict'])
    
    if config.model.freeze_weights:
        mpnn_model.eval()
        # freeze these weights for transfer learning
        for param in mpnn_model.parameters():
            param.requires_grad = False

    return mpnn_model

def get_esm(config):
    """Loading Pre-trained ESM-2 model and tokenizer from Hugging Face"""
    
    # For the 650M model: "facebook/esm2_t33_650M_UR50D"
    hf_model_name = config.get('hf_esm_model_name', "facebook/esm2_t33_650M_UR50D")
    
    # EsmTokenizer replaces the FAIR 'Alphabet'
    tokenizer = EsmTokenizer.from_pretrained(hf_model_name)
    
    # EsmModel
    esm_model = EsmModel.from_pretrained(hf_model_name)
    esm_model = esm_model.to(device)
    
    return esm_model, tokenizer

def mut_seq_generator(wt_seq_dict, topk_aa=None, mutations=None, num_chain=None, protein_type=None):
    """
    Generate a list of tuples with label and mutant sequences based on 
    wild-type sequence and top k amino acid probabilities.
    Args:
        wt_seq_dict: Dictionary of wild-type sequence strings. Key: 'seq_chain_X', Value: sequence
        topk_aa: Optional list of top k amino acid predictions for each position (shape: [L, k]).
        mutations: Optional list of Mutation objects for training mode.
        num_chain: Number of chains in the protein (e.g., 1 for monomer, 2 for homomer, etc.).
        protein_type: Type of protein (e.g., 'monomer', 'homomer', 'heteromer').
    Returns:
        mutated_seqs: List of tuples with label ({WT}{POS}{MUT} / {WT}{chain ID}{POS}{MUT}) and concatenated mutant sequences.
    """
    mutated_seqs = []
    # mut_seq_dict = {}
    # seq_length = len(list(wt_seq_dict.values())[0])
    mut_seq_count = 0
    mutated_seqs.append(('WT', "".join(wt_seq_dict.values())))  # Add the WT sequence at the start

    if mutations is not None:
        if protein_type == 'monomer':
            wt_seq = mutated_seqs[0][1]  # Get the WT sequence for monomer
            for mut in mutations:
                mutated_seq = wt_seq[:mut.position] + mut.mutation + wt_seq[mut.position+1:]
                mutated_seqs.append((f'{mut.wildtype}{str(mut.position)}{mut.mutation}', mutated_seq))
            return mutated_seqs  

        elif protein_type == 'homomer':
            wt_seq = mutated_seqs[0][1]  # Get the concatenated WT homomer sequence 
            for mut in mutations:
                monomer_seq = wt_seq[:len(wt_seq)//num_chain]
                mutated_seq = monomer_seq[:mut.position] + mut.mutation + monomer_seq[mut.position+1:]
                mutated_seqs.append((f'{mut.wildtype}{str(mut.position)}{mut.mutation}', mutated_seq*num_chain))
            return mutated_seqs

        elif protein_type == 'heteromer':
            for mut in mutations:
                full_mutated_seq = ""
                mut_seq = wt_seq_dict[f'seq_chain_{mut.chain}']  # Get the WT heteromer sequence on which mutation occurs
                mutated_seq = mut_seq[:mut.position] + mut.mutation + mut_seq[mut.position+1:]
                for chain_id, seq in wt_seq_dict.items():
                    if chain_id == f'seq_chain_{mut.chain}':
                        full_mutated_seq += mutated_seq
                    else:
                        full_mutated_seq += seq
                mutated_seqs.append((f'{mut.wildtype}{mut.chain}{str(mut.position)}{mut.mutation}', full_mutated_seq))
            return mutated_seqs

        else:
            raise ValueError("Invalid protein type. Must be 'monomer', 'homomer', or 'heteromer'.")
          
    # if topk_aa is not None:
    #     for i in range(seq_length):
    #         for aa in topk_aa[i]:
    #             if aa != wt_seq[i]:  # Avoid adding the WT amino acid
    #                 mutated_seq = wt_seq[:i] + aa + wt_seq[i+1:]
    #                 # if i in mut_seq_dict:
    #                 #     mut_seq_dict[i].append(mutated_seq)
    #                 # else:
    #                 #     mut_seq_dict[i] = [mutated_seq]
    #                 mut_seq_count += 1
    #                 mutated_seqs.append((wt_seq[i]+str(i)+aa, mutated_seq))
    
    # print(f'Total mutated sequences: {mut_seq_count}')
        # return mutated_seqs

def mpnn_score_generator(aa_prob, mut_list, wt_seq_dict, protein_type=None, num_chain=None, chain_offsets=None):
    """Generate ProteinMPNN scores for mutant & WT sequences based on amino acid probabilities.
    Returns:
        mut_scores: List of ProteinMPNN scores for each mutant sequence. Shape:[M].
        wt_scores: List of ProteinMPNN scores for each WT amino acid at each position. Shape:[M]."""
    def _parse_label(label):
        """Return the raw mutation token plus an optional chain id."""
        chain_id = None
        token = label

        if ':' in label:
            prefix, token = label.split(':', 1)
            prefix_parts = prefix.split()
            if prefix_parts:
                chain_id = prefix_parts[-1]

        return token, chain_id

    mut_scores = []
    wt_scores = []
    for label, mut_seq in mut_list:
        if label == 'WT':
            continue
        if protein_type == 'monomer':
            token, _ = _parse_label(label)
            wt_aa = token[0]
            position = int(token[1:-1])
            mutant_aa = token[-1]
            wt_index = ALPHABET.index(wt_aa)
            mut_index = ALPHABET.index(mutant_aa)
            wt_score = aa_prob[position, wt_index]
            mut_score = aa_prob[position, mut_index]
            wt_scores.append(wt_score)
            mut_scores.append(mut_score)
        elif protein_type == 'homomer':
            token, _ = _parse_label(label)
            wt_aa = token[0]
            position = int(token[1:-1])
            mutant_aa = token[-1]
            wt_index = ALPHABET.index(wt_aa)
            mut_index = ALPHABET.index(mutant_aa)
            wt_score = 1.0
            mut_score = 1.0
            for chain_id, offset in chain_offsets.items():
                wt_score *= aa_prob[position + offset, wt_index]
                mut_score *= aa_prob[position + offset, mut_index]
            wt_scores.append(wt_score)
            mut_scores.append(mut_score)
        else:  # heteromer 
            token, chain_id = _parse_label(label)
            wt_aa = token[0]
            if chain_id is None:
                chain_id = token[1]
                position = int(token[2:-1])
            else:
                position = int(token[1:-1])
            mutant_aa = token[-1]
            wt_aa = wt_seq_dict[f'seq_chain_{chain_id}'][position]
            assert wt_aa == token[0], f"WT amino acid mismatch: expected {token[0]}, found {wt_aa} in chain {chain_id} at position {position}"
            wt_index = ALPHABET.index(wt_aa)
            mut_index = ALPHABET.index(mutant_aa)
            # Adjust the position index for the concatenated sequence in heteromers
            correct_position = 0
            for ch_id, seq in wt_seq_dict.items():
                if ch_id == f'seq_chain_{chain_id}':
                    correct_position += position
                    break
                else:
                    correct_position += len(seq)
            wt_score = aa_prob[correct_position, wt_index]
            mut_score = aa_prob[correct_position, mut_index]
            wt_scores.append(wt_score)
            mut_scores.append(mut_score)

    return mut_scores, wt_scores

## Structure-Informed Sequence Embedding Model ##
class CrossAttentionLayer(nn.Module):
    """Cross-attention layer to combine ProteinMPNN and ESM2 embeddings"""
    def __init__(self, 
                 query_dim = 1280,      # ESM2 embedding size
                 context_dim = 128,     # ProteinMPNN embedding size
                 hidden_dim = 1024,      # Projected dimension for attention 
                 num_heads = 8, 
                 dropout = 0.1,
                 use_lora = False,
                 lora_rank = 0,
                 lora_alpha = 1.0,
                 lora_dropout = 0.0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_lora = bool(use_lora)
        self.lora_rank = int(lora_rank)
        
        # Linear projections for query, key, value (Weight matrices)
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(context_dim, hidden_dim)
        self.value_proj = nn.Linear(context_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

        if self.use_lora and self.lora_rank > 0:
            self.query_proj = LoRALinear(query_dim, hidden_dim, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.key_proj = LoRALinear(context_dim, hidden_dim, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.value_proj = LoRALinear(context_dim, hidden_dim, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

            self.out_proj = LoRALinear(hidden_dim, query_dim, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.norm = nn.LayerNorm(query_dim) # layer normalisation

        if self.use_lora and self.lora_rank > 0:
            self.query_proj.freeze_base_weights()
            self.key_proj.freeze_base_weights()
            self.value_proj.freeze_base_weights()
            self.out_proj.freeze_base_weights()
        
    def forward(self, mpnn_emb: torch.Tensor, esm_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            esm_emb: Query embeddings from ESM2 [batch_size, seq_len, embedding_dim]
            mpnn_emb: Key/Value embeddings from ProteinMPNN [batch_size, seq_len, embedding_dim]
        Returns:
            Cross-attention output [batch_size, seq_len, embedding_dim]
        """
        batch_size, Lq, _ = esm_emb.shape
        _, Lk, _ = mpnn_emb.shape

        residual = esm_emb  # for residual connection
        
        Q = self.query_proj(esm_emb)   # Q=X*W_q [batch_size, seq_len, embedding_dim]
        K = self.key_proj(mpnn_emb)      # K=Y*W_k [batch_size, seq_len, embedding_dim]
        V = self.value_proj(mpnn_emb)    # V=Y*W_v [batch_size, seq_len, embedding_dim]
        
        # Reshape the hidden dim for multi-head attention ([B, N, H, D_h] -> [B, H, N, D_h])
        Q = Q.view(batch_size, Lq, self.num_heads, self.head_dim).transpose(1, 2) # shape: [batch_size, num_heads, Lq=1, head_dim]
        K = K.view(batch_size, Lk, self.num_heads, self.head_dim).transpose(1, 2) # shape: [batch_size, num_heads, Lk=90, head_dim]
        V = V.view(batch_size, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5) # [B, num_heads, Lq, Lk]
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V) # [B, H, Lq, Dh]
        
        # Reshape back
        context = context.transpose(1, 2).contiguous() 
        context = context.view(batch_size, Lq, self.hidden_dim) # reshape to [B, Lq, hidden_dim]
        
        # Output projection
        output = self.out_proj(context) # shape: [batch_size, seq_len, embedding_dim]
        output = self.dropout(output) 
        
        return self.norm(output + residual)
        # return output

class StabilityPredictionModel(nn.Module):
    """Stability prediction model combining ProteinMPNN and ESM2 embeddings"""
    def __init__(self, config,
                 mpnn_dim = 128, 
                 esm_dim = 1280, 
                 mpnn_score_dim = 64,
                 dropout = 0.3):
        super().__init__()

        self.config = config
        self.mpnn_dim = mpnn_dim
        self.esm_dim = esm_dim
        self.attn_hidden_dim = config.model.attn_hidden_dim
        self.num_heads = config.model.num_heads
        self.attn_dropout = config.model.attn_dropout
        self.mlp_dropout = config.model.mlp_dropout
        self.finetune_mode = str(getattr(config.finetuning, "mode", "mlp_only")).lower()
        self.use_lora_cross_attention = bool(getattr(config.finetuning, "use_lora_cross_attention", False)) or self.finetune_mode == "lora_mlp"
        self.lora_rank = int(getattr(config.finetuning, "lora_rank", 0))
        self.lora_alpha = float(getattr(config.finetuning, "lora_alpha", max(1, self.lora_rank)))
        self.lora_dropout = float(getattr(config.finetuning, "lora_dropout", 0.0))

        # Calling pre-trained ProteinMPNN model
        self.mpnn_model = get_protein_mpnn(config) 
        # Calling pre-trained ESM-2 model and tokenizer
        self.esm_model, self.esm_tokenizer = get_esm(config) 
        self.esm_model = self.esm_model.half().to(device)
        # Cross-Attention layer
        # self.cross_attention = CrossAttentionLayer(query_dim=self.esm_dim, context_dim=self.mpnn_dim, hidden_dim=self.attn_hidden_dim, num_heads=self.num_heads, dropout=self.attn_dropout)
        self.cross_attention = CrossAttentionLayer(
            query_dim=self.esm_dim,
            context_dim=self.mpnn_dim,
            hidden_dim=self.attn_hidden_dim,
            num_heads=self.num_heads,
            dropout=self.attn_dropout,
            use_lora=self.use_lora_cross_attention,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
        )
        self._mpnn_feature_cache = {}

        # Project log_diff as an explicit feature rather than a gate
        self.mpnn_score_proj = nn.Sequential(
            nn.Linear(1, mpnn_score_dim),
            nn.LayerNorm(mpnn_score_dim),
            nn.GELU(),
            nn.Linear(mpnn_score_dim, mpnn_score_dim),
        )
        
        # MLP updated architecture
        input_dim = 3 * esm_dim + 64  # MLP input: 3*1280 + 64 = 3904
        self.mlp = nn.Sequential(
        # First Layer: Wide to capture initial interactions
        nn.LayerNorm(input_dim), 
        nn.Linear(input_dim, 1024),
        nn.LayerNorm(1024), # Added for training stability
        nn.GELU(),
        nn.Dropout(self.mlp_dropout),
        
        # Second Layer: Compressing information
        nn.Linear(1024, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(self.mlp_dropout),
        
        # Third Layer: Final mapping to stability
        nn.Linear(256, 64),
        nn.LayerNorm(64),
        nn.Dropout(self.mlp_dropout),
        nn.GELU(),
        
        # Output: Single scalar (normalized ddG in [-1, 1])
        nn.Linear(64, 1),
    )

    def _forward_from_cached_features(self, cached_features, mutations=None, mut_list=None, num_chains=None):
        """Run the lightweight prediction head using precomputed embeddings."""
        wt_seq_dict = cached_features["wt_seq_dict"]
        pdb_type = cached_features["pdb_type"]
        chain_ids = cached_features["chain_ids"]
        chain_sequences = cached_features["chain_sequences"]
        chain_offsets = cached_features["chain_offsets"]
        chain_lengths = cached_features["chain_lengths"]
        residue_embeddings = cached_features["residue_embeddings"]
        names_out = cached_features["names_out"]
        mpnn_layer_embed = cached_features["mpnn_layer_embed"]
        mut_scores = cached_features["mut_scores"]
        wt_scores = cached_features["wt_scores"]

        if torch.is_tensor(residue_embeddings):
            residue_embeddings = residue_embeddings.to(device)
        else:
            residue_embeddings = torch.stack([t.to(device) for t in residue_embeddings], dim=0)

        if torch.is_tensor(mpnn_layer_embed):
            mpnn_layer_embed = mpnn_layer_embed.to(device)
        else:
            mpnn_layer_embed = torch.as_tensor(mpnn_layer_embed, device=device)

        mut_scores = torch.as_tensor(mut_scores, device=device, dtype=mpnn_layer_embed.dtype)
        wt_scores = torch.as_tensor(wt_scores, device=device, dtype=mpnn_layer_embed.dtype)

        try:
            wt_index = names_out.index('WT')
        except ValueError:
            wt_index = 0

        mutant_indices = []
        mutant_positions = []
        mutant_is_homomer = []

        for i, label in enumerate(names_out):
            if label == 'WT':
                continue
            if pdb_type == 'monomer':
                position = int(label[1:-1])
                mutant_indices.append(i)
                mutant_positions.append(position)
                mutant_is_homomer.append(False)
            elif pdb_type == 'homomer':
                position = int(label[1:-1])
                all_chain_positions = [
                    chain_offsets[cid] + position
                    for cid in chain_ids
                    if position < chain_lengths[cid]
                ]
                mutant_indices.append(i)
                mutant_positions.append(all_chain_positions)
                mutant_is_homomer.append(True)
            else:
                chain_id = label[1]
                position = int(label[2:-1])
                concat_position = chain_offsets[chain_id] + position
                mutant_indices.append(i)
                mutant_positions.append(concat_position)
                mutant_is_homomer.append(False)

        if len(mutant_indices) == 0:
            return torch.empty(0, device=device), torch.empty(0, device=device)

        prediction_batch_size = int(getattr(self.config.model, "prediction_batch_size", 64))
        all_ddg_fwd = []
        all_ddg_rev = []
        mpnn_layer = 0

        for batch_start in range(0, len(mutant_indices), prediction_batch_size):
            batch_end = min(batch_start + prediction_batch_size, len(mutant_indices))
            batch_mut_indices = mutant_indices[batch_start:batch_end]
            batch_mut_positions = mutant_positions[batch_start:batch_end]

            mpnn_kv_list = []
            esm_mut_q_list = []
            esm_wt_q_list = []

            k = mpnn_layer_embed.unsqueeze(0)
            for pos, seq_idx in zip(batch_mut_positions, batch_mut_indices):
                q_mut = residue_embeddings[seq_idx].to(dtype=mpnn_layer_embed.dtype).unsqueeze(0)
                q_wt = residue_embeddings[wt_index].to(dtype=mpnn_layer_embed.dtype).unsqueeze(0)
                mpnn_kv_list.append(k)
                esm_mut_q_list.append(q_mut)
                esm_wt_q_list.append(q_wt)

            mpnn_batch = torch.cat(mpnn_kv_list, dim=0)
            esm_mutant_batch = torch.cat(esm_mut_q_list, dim=0)
            esm_wt_batch = torch.cat(esm_wt_q_list, dim=0)

            delta_esm_fwd = esm_mutant_batch - esm_wt_batch
            delta_esm_rev = -delta_esm_fwd

            attn_fwd = self.cross_attention(mpnn_batch, delta_esm_fwd)
            attn_rev = self.cross_attention(mpnn_batch, delta_esm_rev)

            batch_size = len(batch_mut_indices)
            attn_fwd_pos_list = []
            attn_rev_pos_list = []
            esm_mut_pos_list = []
            esm_wt_pos_list = []

            for b_i, (seq_idx, pos) in enumerate(zip(batch_mut_indices, batch_mut_positions)):
                is_homo = mutant_is_homomer[batch_start + b_i]
                if is_homo:
                    fwd_vecs = attn_fwd[b_i, pos, :]
                    rev_vecs = attn_rev[b_i, pos, :]
                    mut_vecs = esm_mutant_batch[b_i, pos, :]
                    wt_vecs = esm_wt_batch[b_i, pos, :]
                    attn_fwd_pos_list.append(fwd_vecs.mean(dim=0))
                    attn_rev_pos_list.append(rev_vecs.mean(dim=0))
                    esm_mut_pos_list.append(mut_vecs.mean(dim=0))
                    esm_wt_pos_list.append(wt_vecs.mean(dim=0))
                else:
                    attn_fwd_pos_list.append(attn_fwd[b_i, pos, :])
                    attn_rev_pos_list.append(attn_rev[b_i, pos, :])
                    esm_mut_pos_list.append(esm_mutant_batch[b_i, pos, :])
                    esm_wt_pos_list.append(esm_wt_batch[b_i, pos, :])

            attn_fwd_pos = torch.stack(attn_fwd_pos_list, dim=0).unsqueeze(1)
            attn_rev_pos = torch.stack(attn_rev_pos_list, dim=0).unsqueeze(1)
            esm_mut_pos = torch.stack(esm_mut_pos_list, dim=0).unsqueeze(1)
            esm_wt_pos = torch.stack(esm_wt_pos_list, dim=0).unsqueeze(1)

            mut_scores_batch = mut_scores[batch_start:batch_end].contiguous().view(batch_size, 1, 1)
            wt_scores_batch = wt_scores[batch_start:batch_end].contiguous().view(batch_size, 1, 1)

            delta_scores_fwd = torch.log(wt_scores_batch) - torch.log(mut_scores_batch)
            delta_scores_rev = -delta_scores_fwd
            proj_delta_scores_fwd = self.mpnn_score_proj(delta_scores_fwd)
            proj_delta_scores_rev = self.mpnn_score_proj(delta_scores_rev)
            attn_fwd_pos = torch.cat([attn_fwd_pos, proj_delta_scores_fwd], dim=-1)
            attn_rev_pos = torch.cat([attn_rev_pos, proj_delta_scores_rev], dim=-1)

            comb_fwd = torch.cat([esm_wt_pos, esm_mut_pos, attn_fwd_pos], dim=-1)
            comb_rev = torch.cat([esm_mut_pos, esm_wt_pos, attn_rev_pos], dim=-1)
            ddg_fwd = self.mlp(comb_fwd).view(-1)
            ddg_rev = self.mlp(comb_rev).view(-1)

            all_ddg_fwd.append(ddg_fwd)
            all_ddg_rev.append(ddg_rev)

        ddg_fwd_final = torch.cat(all_ddg_fwd, dim=0)
        ddg_rev_final = torch.cat(all_ddg_rev, dim=0)
        return ddg_fwd_final, ddg_rev_final

    def _cache_payload(self, pdb_type, chain_ids, chain_sequences, chain_offsets, chain_lengths, wt_seq_dict, names_out, residue_embeddings, mpnn_layer_embed, mut_scores, wt_scores):
        """Pack precomputed tensors for disk caching."""
        if isinstance(residue_embeddings, list):
            residue_embeddings = torch.stack([t.detach().cpu() for t in residue_embeddings], dim=0)
        else:
            residue_embeddings = residue_embeddings.detach().cpu()

        return {
            "pdb_type": pdb_type,
            "chain_ids": list(chain_ids),
            "chain_sequences": list(chain_sequences),
            "chain_offsets": dict(chain_offsets),
            "chain_lengths": dict(chain_lengths),
            "wt_seq_dict": {k: v for k, v in wt_seq_dict.items()},
            "names_out": list(names_out),
            "residue_embeddings": residue_embeddings,
            "mpnn_layer_embed": mpnn_layer_embed.detach().cpu(),
            "mut_scores": mut_scores.detach().cpu(),
            "wt_scores": wt_scores.detach().cpu(),
        }
        
    def forward(self, pdb, mutations=None, mut_list=None, pdb_type=None, num_chains=None) -> torch.Tensor:
        """
        Forward pass for the stability prediction model.
        Args:
            pdb: pdb object of batch size = 1 containing the structure and sequence information of the target protein complex.
            mutations: list of Mutation objects
            mut_list: list of tuples (label, mutant sequence)
            pdb_type: type of the protein (monomer, homomer, heteromer)
            num_chains: number of chains in the PDB structure

        Returns:
            ddG predictions [batch_size, seq_len]
        """
        cache_record = pdb[0].get("cached_features") if isinstance(pdb[0], dict) else None
        if cache_record is not None:
            return self._forward_from_cached_features(cache_record, mutations=mutations, mut_list=mut_list, num_chains=num_chains)

        ## Generate MPNN structure embeddings ##
        pdb_record = pdb[0]
        cache_key = self._get_mpnn_cache_key(pdb_record)
        cached_features = self._mpnn_feature_cache.get(cache_key)
        if cached_features is None:
            cached_features = tied_featurize([pdb_record], device, None, None, None, None, None, None, ca_only=False)
            self._mpnn_feature_cache[cache_key] = _move_nested_tensors(cached_features, torch.device("cpu"))

        cached_features = _move_nested_tensors(cached_features, device)

        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = cached_features
        log_probs, mpnn_enc_embeds, mpnn_dec_embeds, mpnn_seq_embeds = self.mpnn_model(X, S, mask, chain_M, residue_idx, chain_encoding_all, None) # mpnn_embeds shape: [B, L, 128]
        mpnn_enc_embeds = [t.to(device) for t in mpnn_enc_embeds] # move every tensor element to cuda device # shape: [num_layers][B, L, 128]
        mpnn_dec_embeds = [t.to(device) for t in mpnn_dec_embeds] # move every tensor element to cuda device # shape: [num_layers][B, L, 128]
        mpnn_seq_embeds = mpnn_seq_embeds.to(device) # shape: [B, L, 128]
        wt_seq_dict = {key: pdb[0][key] for key in pdb[0].keys() if key.startswith('seq_chain')}

        # Resolve chain metadata once; shared helper uses this schema for oligomer token prep.
        # wt_chain_sequences, chain_keys = get_ordered_chain_sequences(pdb_record)
        pdb_type, chain_ids, chain_sequences = detect_oligomer_type(pdb_record)
        
        # Build chain offset map: {chain_id: start_index_in_concat_seq}
        chain_offsets = {}
        offset = 0
        for chain_id, chain_seq in zip(chain_ids, chain_sequences):
            chain_offsets[chain_id] = offset
            offset += len(chain_seq)
        chain_lengths = {chain_id: len(seq) for chain_id, seq in zip(chain_ids, chain_sequences)}

        # Auto-resolve missing metadata so training callers can pass only (pdb, mutations).
        if num_chains is None:
            num_chains = len(chain_ids)
            assert num_chains == pdb_record['num_of_chains']

        ## Generate mutant sequences ##
        aa_prob = np.exp(log_probs[0].detach().cpu().numpy())  # Convert log_probs to probabilities. shape: [L, C=21]
        alpha_dict = {0:'A', 1:'C', 2:'D', 3:'E', 4:'F', 5:'G', 6:'H', 7:'I', 8:'K', 9:'L', 10:'M', 11:'N',  
                 12:'P', 13:'Q', 14:'R', 15:'S', 16:'T', 17:'V', 18:'W', 19:'Y', 20:'X', 21:'-'}
        
        if self.config.training.train_flag and mutations is not None:
            mut_list = mut_seq_generator(wt_seq_dict, mutations=mutations, protein_type=pdb_type, num_chain=num_chains)  # List of tuples of (label, mutant sequence)
            ## Generate MPNN scores for mutant sequences ##
            mut_scores, wt_scores = mpnn_score_generator(aa_prob, mut_list, wt_seq_dict, protein_type=pdb_type, num_chain=num_chains, chain_offsets=chain_offsets)
            mut_scores = torch.tensor(mut_scores, device=device, dtype=mpnn_dec_embeds[0].dtype)
            wt_scores = torch.tensor(wt_scores, device=device, dtype=mpnn_dec_embeds[0].dtype)

        elif mut_list is not None:
            mut_list = mut_list
            ## Generate MPNN scores for mutant sequences ##
            mut_scores, wt_scores = mpnn_score_generator(aa_prob, mut_list, wt_seq_dict, protein_type=pdb_type, num_chain=num_chains, chain_offsets=chain_offsets)
            mut_scores = torch.tensor(mut_scores, device=device, dtype=mpnn_dec_embeds[0].dtype)
            wt_scores = torch.tensor(wt_scores, device=device, dtype=mpnn_dec_embeds[0].dtype)
            
        else:
            print("No mutations or mutant list provided.")
            # k = self.config.topk_mutations # top k mutations
            # topk_indices = np.argsort(aa_prob, axis=1)[:,-k:] # top k indices for each row
            # topk_indices = np.flip(topk_indices, axis=1)  # flip to have highest prob first
            # topk_aa = np.vectorize(alpha_dict.get)(topk_indices)
            # mut_list = mut_seq_generator(wt_seq_dict, topk_aa=topk_aa, protein_type=pdb_type, num_chain=num_chains)  # List of tuples of (label, mutant sequence)

        # Ensure WT is present as baseline for delta embeddings.
        if not any(label == 'WT' for label, _ in mut_list):
            mut_list = [('WT', "".join(wt_seq_dict.values()))] + mut_list
            
        ## Generate ESM2 sequence embeddings ##
        self.esm_model = self.esm_model.eval() 

        if pdb_type != 'monomer':
            use_oligomer_esm = bool(getattr(self.config.model, "use_oligomer_esm", True))
        else:
            use_oligomer_esm = False

        # helper fn for dynamic batching (original training-friendly path)
        def batched(iterable, batch_size):
            for i in range(0, len(iterable), batch_size):
                yield iterable[i : i + batch_size]


        batch_size = int(getattr(self.config.model, "esm_batch_size", 16))

        # Store per-sequence embeddings and optionally per-residue arrays
        # sequence_embeddings = []
        residue_embeddings = []  # list of np arrays (L x C)
        names_out = [] # list of sequence labels

        layer_idx = self.esm_model.config.num_hidden_layers  # final layer index (0-indexed) for HF ESM

        require_position_ids = bool(getattr(self.config.model, "require_oligomer_position_ids", True))
        esm_forward_params = inspect.signature(self.esm_model.forward).parameters
        supports_position_ids = "position_ids" in esm_forward_params

        if use_oligomer_esm and require_position_ids and not supports_position_ids:
            raise RuntimeError(
                "Oligomer ESM path requires custom position_ids, but current ESM backend "
                "does not support a position_ids argument in forward(). "
                "Use an ESM implementation that supports position_ids, or set "
                "model.require_oligomer_position_ids=False to allow fallback behavior."
            )

        if use_oligomer_esm:
            oligomer_batch_size = int(getattr(self.config.model, "esm_batch_size", 8))
            for batch_start in range(0, len(mut_list), oligomer_batch_size):
                batch = mut_list[batch_start:batch_start + oligomer_batch_size]
                batch_labels = []
                batch_tokens = []
                batch_position_ids = []

                for label, seq in batch:
                    tokens, position_ids = build_oligomer_tokens(
                        pdb_record,
                        self.esm_tokenizer,
                        gap_size=int(getattr(self.config.model, "esm_chain_gap", 100)),
                        full_seq=seq,
                    )
                    batch_labels.append(label)
                    batch_tokens.append(tokens.squeeze(0))
                    batch_position_ids.append(position_ids.squeeze(0))

                try:
                    tokens = nn.utils.rnn.pad_sequence(
                        batch_tokens,
                        batch_first=True,
                        padding_value=self.esm_tokenizer.pad_token_id,
                    ).to(device)
                    position_ids = nn.utils.rnn.pad_sequence(
                        batch_position_ids,
                        batch_first=True,
                        padding_value=0,
                    ).to(device)
                    attention_mask = (tokens != self.esm_tokenizer.pad_token_id).to(device)

                    with torch.no_grad():
                        # Batched HuggingFace ESM forward with position_ids
                        out = self.esm_model(
                            input_ids=tokens,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            output_hidden_states=True,
                        )
                        token_reps = out.last_hidden_state

                        padding_idx = self.esm_tokenizer.pad_token_id
                        cls_idx = self.esm_tokenizer.cls_token_id
                        eos_idx = self.esm_tokenizer.eos_token_id

                        valid_mask = (tokens != padding_idx)
                        valid_mask &= (tokens != cls_idx)
                        valid_mask &= (tokens != eos_idx)

                        for i, label in enumerate(batch_labels):
                            names_out.append(label)
                            res_emb = token_reps[i][valid_mask[i]]
                            residue_embeddings.append(res_emb)

                except RuntimeError as e:
                    print("RuntimeError during model forward:", e)
                    print("Try reducing oligomer_batch_size, chain count, or ESM precision settings and re-run.")
                    torch.cuda.empty_cache()
                    raise
        else:
            # Original monomer-style/batched path (default) to preserve existing training behavior.
            for batch in batched(mut_list, batch_size):
                # Tokenize sequences with HuggingFace tokenizer
                batch_labels = [label for label, seq in batch]
                batch_seqs = [seq for label, seq in batch]
                # Tokenizer returns dict with 'input_ids' and optional 'attention_mask'
                encoded = self.esm_tokenizer(batch_seqs, return_tensors="pt", padding=True)
                tokens = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                try:
                    with torch.no_grad():
                        out = self.esm_model(
                            input_ids=tokens,
                            attention_mask=attention_mask,
                            output_hidden_states=True
                        )
                        token_reps = out.hidden_states[layer_idx]  # +1 for same reason as above

                        padding_idx = self.esm_tokenizer.pad_token_id
                        cls_idx = self.esm_tokenizer.cls_token_id
                        eos_idx = self.esm_tokenizer.eos_token_id

                        valid_mask = (tokens != padding_idx)
                        valid_mask &= (tokens != cls_idx)
                        valid_mask &= (tokens != eos_idx)

                        for i, (label, seq) in enumerate(batch):
                            names_out.append(label)
                            res_emb = token_reps[i][valid_mask[i]]
                            residue_embeddings.append(res_emb)

                except RuntimeError as e:
                    print("RuntimeError during model forward:", e)
                    print("Try lowering batch_size and re-running. Current batch_size was", batch_size)
                    torch.cuda.empty_cache()
                    raise
            

        ## Predict ddG values for mutants ##
        # Process predictions in batches to avoid OOM when stacking all embeddings at once
        if len(residue_embeddings) == 0:
            raise RuntimeError("No ESM residue embeddings were collected - check batching or input sequences")

        # Prepare lists of sequence labels and mutant info in the same order as residue_embeddings
        seq_labels = [t[0] for t in mut_list]
        
        try:
            wt_index = names_out.index('WT')
        except ValueError:
            wt_index = 0

        mutant_indices = []
        mutant_positions = [] # mutated residue positions
        mutant_is_homomer = []  # bool per mutant
        
        for i, label in enumerate(names_out):
            if label == 'WT':
                continue
            if pdb_type == 'monomer': 
                wt_aa = label[0]
                position = int(label[1:-1])
                mutant_aa = label[-1]
                mutant_indices.append(i)
                mutant_positions.append(position)
                mutant_is_homomer.append(False)
            elif pdb_type == 'homomer':
                wt_aa = label[0]
                position = int(label[1:-1])
                mutant_aa = label[-1]
                chain_len = chain_lengths[chain_ids[0]]
                # Collect the same relative position across every chain copy
                all_chain_positions = [
                    chain_offsets[cid] + position
                    for cid in chain_ids
                    if position < chain_lengths[cid]   # safety guard
                ]
                mutant_indices.append(i)
                mutant_positions.append(all_chain_positions)
                mutant_is_homomer.append(True)
            else: # heteromer
                wt_aa = label[0]
                chain_id = label[1]
                position = int(label[2:-1])
                mutant_aa = label[-1]
                # Map chain-local position → position in concatenated sequence
                concat_position = chain_offsets[chain_id] + position
                mutant_indices.append(i)
                mutant_positions.append(concat_position)
                mutant_is_homomer.append(False)

        if len(mutant_indices) == 0:
            return torch.empty(0, device=device), torch.empty(0, device=device)

        # Accumulate predictions in batches
        prediction_batch_size = int(getattr(self.config.model, "prediction_batch_size", 64))
        all_ddg_fwd = []
        all_ddg_rev = []
        
        mpnn_layer = 2 # MPNN encoder layer for embeddings extraction
        
        # Process mutants in batches to keep memory usage bounded
        for batch_start in range(0, len(mutant_indices), prediction_batch_size):
            batch_end = min(batch_start + prediction_batch_size, len(mutant_indices))
            batch_mut_indices = mutant_indices[batch_start:batch_end]
            batch_mut_positions = mutant_positions[batch_start:batch_end]
            
            # Build tensors only for this batch
            mpnn_kv_list = []
            esm_mut_q_list = []
            esm_wt_q_list = []
            
            mpnn_seq_embeds = mpnn_seq_embeds.to(dtype=mpnn_dec_embeds[0].dtype)  # [1, L, mpnn_dim]
            for pos, seq_idx in zip(batch_mut_positions, batch_mut_indices):
                # MPNN (WT) key, value vector for the whole WT sequence 
                k = mpnn_dec_embeds[mpnn_layer][0, :, :].unsqueeze(0)  # [1,L,mpnn_dim]
                # ESM sequence embeddings (convert to target dtype)
                q_mut = residue_embeddings[seq_idx].to(dtype=mpnn_dec_embeds[0].dtype).unsqueeze(0)  # [1,L,esm_dim]
                q_wt = residue_embeddings[wt_index].to(dtype=mpnn_dec_embeds[0].dtype).unsqueeze(0)   # [1,L,esm_dim]
                
                mpnn_kv_list.append(k)
                esm_mut_q_list.append(q_mut)
                esm_wt_q_list.append(q_wt)
            
            # Concatenate batch
            mpnn_batch = torch.cat(mpnn_kv_list, dim=0)  # [batch_size, L, mpnn_dim*2]
            esm_mutant_batch = torch.cat(esm_mut_q_list, dim=0)  # [batch_size, L, esm_dim]
            esm_wt_batch = torch.cat(esm_wt_q_list, dim=0)  # [batch_size, L, esm_dim]

            # Run cross-attention on MPNN and delta ESM embeddings
            delta_esm_fwd = esm_mutant_batch - esm_wt_batch  # [batch_size, L, esm_dim]
            delta_esm_rev = -delta_esm_fwd  # [batch_size, L, esm_dim]
            
            attn_fwd = self.cross_attention(mpnn_batch, delta_esm_fwd)  # [batch_size, L, esm_dim]
            attn_rev = self.cross_attention(mpnn_batch, delta_esm_rev)  # [batch_size, L, esm_dim]
            
            # Extract embeddings at the correct mutant position(s)
            batch_size = len(batch_mut_indices)
            attn_fwd_pos_list = []
            attn_rev_pos_list = []
            esm_mut_pos_list  = []
            esm_wt_pos_list   = []

            for b_i, (seq_idx, pos) in enumerate(zip(batch_mut_indices, batch_mut_positions)):
                is_homo = mutant_is_homomer[batch_start + b_i]

                if is_homo:
                    # pos is a list of concat positions — one per chain copy
                    # mean-pool attention outputs across all chain copies
                    fwd_vecs = attn_fwd[b_i, pos, :]          # [num_copies, esm_dim]
                    rev_vecs = attn_rev[b_i, pos, :]
                    mut_vecs = esm_mutant_batch[b_i, pos, :]
                    wt_vecs  = esm_wt_batch[b_i, pos, :]

                    attn_fwd_pos_list.append(fwd_vecs.mean(dim=0))   # [esm_dim]
                    attn_rev_pos_list.append(rev_vecs.mean(dim=0))
                    esm_mut_pos_list.append(mut_vecs.mean(dim=0))
                    esm_wt_pos_list.append(wt_vecs.mean(dim=0))
                else:
                    # pos is a single int (monomer or heteromer concat offset)
                    attn_fwd_pos_list.append(attn_fwd[b_i, pos, :])  # [esm_dim]
                    attn_rev_pos_list.append(attn_rev[b_i, pos, :])
                    esm_mut_pos_list.append(esm_mutant_batch[b_i, pos, :])
                    esm_wt_pos_list.append(esm_wt_batch[b_i, pos, :])

            # Stack back to [batch_size, esm_dim], then add dim for concat
            attn_fwd_pos  = torch.stack(attn_fwd_pos_list,  dim=0).unsqueeze(1)  # [B, 1, esm_dim]
            attn_rev_pos  = torch.stack(attn_rev_pos_list,  dim=0).unsqueeze(1)
            esm_mut_pos = torch.stack(esm_mut_pos_list,  dim=0).unsqueeze(1)
            esm_wt_pos     = torch.stack(esm_wt_pos_list,   dim=0).unsqueeze(1)  # [B, 1, esm_dim]
            
            # Concatenating MPNN scores to the attention outputs (only during training)
            # Extract scores for this batch
            mut_scores_batch = mut_scores[batch_start:batch_end].contiguous().view(batch_size, 1, 1) # [B, 1, 1]
            wt_scores_batch = wt_scores[batch_start:batch_end].contiguous().view(batch_size, 1, 1) # [B, 1, 1]

            delta_scores_fwd = torch.log(wt_scores_batch) - torch.log(mut_scores_batch)  # log(WT/Mut) for forward pass
            delta_scores_rev = -delta_scores_fwd
            proj_delta_scores_fwd = self.mpnn_score_proj(delta_scores_fwd)  # Projected MPNN score; [B, 1, 64]
            proj_delta_scores_rev = self.mpnn_score_proj(delta_scores_rev)
            attn_fwd_pos = torch.cat([attn_fwd_pos, proj_delta_scores_fwd], dim=-1)  # [B, 1, esm_dim + 64]
            attn_rev_pos = torch.cat([attn_rev_pos, proj_delta_scores_rev], dim=-1)  # [B, 1, esm_dim + 64]
            
            ## Enforcing symmetry ##
            # --- FORWARD PASS (WT -> Mut) ---
            # comb_fwd = torch.cat([esm_wt_pos, esm_mut_pos, attn_fwd_pos], dim=-1)
            comb_fwd = torch.cat([esm_wt_pos, esm_mut_pos, attn_fwd_pos], dim=-1) # [B, 1, 3*esm_dim + 64]
            ddg_fwd = self.mlp(comb_fwd).view(-1)
            
            # --- REVERSE PASS (Mut -> WT) ---
            # comb_rev = torch.cat([esm_mut_pos, esm_wt_pos, attn_rev_pos], dim=-1)
            comb_rev = torch.cat([esm_mut_pos, esm_wt_pos, attn_rev_pos], dim=-1) # [B, 1, 3*esm_dim + 64]
            ddg_rev = self.mlp(comb_rev).view(-1)
            
            all_ddg_fwd.append(ddg_fwd)
            all_ddg_rev.append(ddg_rev)
            
            # Clean up batch tensors to free memory
            del mpnn_batch, esm_mutant_batch, esm_wt_batch, attn_fwd, attn_rev
            torch.cuda.empty_cache()
        
        # Concatenate all batch predictions
        ddg_fwd_final = torch.cat(all_ddg_fwd, dim=0)
        ddg_rev_final = torch.cat(all_ddg_rev, dim=0)
        
        return ddg_fwd_final, ddg_rev_final  # shape: [num_mutants] each

    def _get_mpnn_cache_key(self, pdb_record):
        """Build a stable cache key for structure-only ProteinMPNN features."""
        chain_sequences, chain_keys = get_ordered_chain_sequences(pdb_record)
        return (
            pdb_record.get("name", ""),
            pdb_record.get("seq", ""),
            tuple(chain_keys),
            tuple(chain_sequences),
            pdb_record.get("num_of_chains", len(chain_sequences)),
        )