import torch
from torch.utils.data import ConcatDataset
import pandas as pd
import numpy as np
import pickle
import os
import re
from Bio import pairwise2, SeqIO
from math import isnan
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional

from protein_mpnn_utils import alt_parse_PDB, parse_PDB
from cache import cache


ALPHABET = 'ACDEFGHIKLMNPQRSTVWY-'


@cache(lambda cfg, pdb_file: pdb_file)
def parse_pdb_cached(cfg, pdb_file):
    '''Parse PDB file with caching to speed up repeated access.
    Input: pdb_file: path to PDB file
    Output: list of dicts, one per model in the PDB, with keys '''
    return parse_PDB(pdb_file)


def load_fasta_ids(fasta_path):
    """Load FASTA record IDs into a set."""
    if fasta_path is None:
        return None

    fasta_path = os.fspath(fasta_path)
    ids = set()
    for record in SeqIO.parse(fasta_path, "fasta"):
        ids.add(record.id)
    return ids


def _first_existing_column(df, candidates):
    for column_name in candidates:
        if column_name in df.columns:
            return column_name
    raise KeyError(f"None of the columns {candidates} were found in {list(df.columns)}")


def _clean_benchmark_sequence(seq):
    if pd.isna(seq):
        return ""
    seq = str(seq).strip()
    seq = seq.split(">", 1)[0]
    return re.sub(r"[^A-ZX-]", "", seq.upper())


def _normalize_chain_value(chain_value):
    if pd.isna(chain_value):
        return None
    chain_value = str(chain_value).strip()
    if not chain_value:
        return None
    if "," in chain_value:
        return [part.strip() for part in chain_value.split(",") if part.strip()]
    return chain_value


def _parse_mutation_label(mut_label):
    mut_label = str(mut_label).strip()
    if len(mut_label) < 3:
        raise ValueError(f"Invalid mutation label: {mut_label}")
    wt_aa = mut_label[0].upper()
    mut_aa = mut_label[-1].upper()
    position = int(mut_label[1:-1]) # 1-based indexing
    return wt_aa, position, mut_aa


@dataclass
class Mutation:
    position: int
    wildtype: str
    mutation: str
    ddG: Optional[float] = None
    pdb: Optional[str] = ''
    chain: Optional[str] = None


def seq1_index_to_seq2_index(align, index):
    """Do quick conversion of index after alignment"""
    cur_seq1_index = 0

    # first find the aligned index
    for aln_idx, char in enumerate(align.seqA):
        if char != '-':
            cur_seq1_index += 1
        if cur_seq1_index > index:
            break
    
    # now the index in seq 2 cooresponding to aligned index
    if align.seqB[aln_idx] == '-':
        return None

    seq2_to_idx = align.seqB[:aln_idx+1]
    seq2_idx = aln_idx
    for char in seq2_to_idx:
        if char == '-':
            seq2_idx -= 1
    
    if seq2_idx < 0:
        return None

    return seq2_idx


class MegaScaleDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split, allowed_protein_ids=None, allowed_protein_ids_fasta=None): 

        self.cfg = cfg
        self.split = split  # which split to retrieve

        if allowed_protein_ids is None:
            allowed_protein_ids = getattr(getattr(cfg, "data_loc", {}), "megascale_allowlist_ids", None) 
        if allowed_protein_ids_fasta is None:
            allowed_protein_ids_fasta = getattr(getattr(cfg, "data_loc", {}), "megascale_allowlist_fasta", None)

        if allowed_protein_ids is not None:
            if not isinstance(allowed_protein_ids, set):
                allowed_protein_ids = set(allowed_protein_ids)
        elif allowed_protein_ids_fasta is not None:
            allowed_protein_ids = load_fasta_ids(allowed_protein_ids_fasta)

        fname = self.cfg.data_loc.megascale_csv
        # only load rows needed to save memory
        df = pd.read_csv(fname, usecols=["ddG_ML", "mut_type", "WT_name", "aa_seq", "dG_ML"])
        # remove unreliable data and more complicated mutations
        df = df.loc[df.ddG_ML != '-', :].reset_index(drop=True)
        df = df.loc[~df.mut_type.str.contains("ins") & ~df.mut_type.str.contains("del") & ~df.mut_type.str.contains(":"), :].reset_index(drop=True)

        self.df = df

        # load splits produced by mmseqs clustering
        with open(self.cfg.data_loc.megascale_splits, 'rb') as f:
            splits = pickle.load(f)  # this is a dict with keys train/val/test and items holding FULL PDB names for a given split
            
        self.split_wt_names = {
            "val": [],
            "test": [],
            "train": [],
            "train_s669": [],
            "all": [], 
            "cv_train_0": [],
            "cv_train_1": [],
            "cv_train_2": [],
            "cv_train_3": [],
            "cv_train_4": [],
            "cv_val_0": [],
            "cv_val_1": [],
            "cv_val_2": [],
            "cv_val_3": [],
            "cv_val_4": [],
            "cv_test_0": [],
            "cv_test_1": [],
            "cv_test_2": [],
            "cv_test_3": [],
            "cv_test_4": [],
        }

        if 'reduce' not in cfg:
            cfg.reduce = ''

        self.wt_seqs = {}
        self.mut_rows = {}

        if self.split == 'all':
            all_names = np.concatenate([
                splits['train'],
                splits['val'],
                splits['test']
    ])
            self.split_wt_names[self.split] = all_names
        else:
            if cfg.reduce == 'prot' and self.split == 'train':
                n_prots_reduced = 58
                self.split_wt_names[self.split] = np.random.choice(splits["train"], n_prots_reduced)
            else:
                self.split_wt_names[self.split] = splits[self.split]

        self.wt_names = self.split_wt_names[self.split]

        # Optionally restrict to proteins that appear in a cleaned benchmark allowlist.
        if allowed_protein_ids is not None and self.split == 'train':
            filtered_wt_names = [wt_name for wt_name in self.wt_names if wt_name in allowed_protein_ids]
            removed = len(self.wt_names) - len(filtered_wt_names)
            print(f"MegaScale allowlist filter: kept {len(filtered_wt_names)} / {len(self.wt_names)} proteins")
            if removed > 0:
                print(f"  Removed {removed} proteins not present in the allowlist FASTA")
            self.wt_names = filtered_wt_names

        if len(self.wt_names) == 0:
            raise ValueError(
                "MegaScaleDataset has no proteins after applying the allowlist filter. "
                "Check that your cleaned FASTA IDs match WT_name values in the MegaScale CSV."
            )

        for wt_name in tqdm(self.wt_names):
            wt_rows = df.query('WT_name == @wt_name and mut_type == "wt"').reset_index(drop=True)
            self.mut_rows[wt_name] = df.query('WT_name == @wt_name and mut_type != "wt"').reset_index(drop=True)
            if type(cfg.reduce) is float and self.split == 'train':
                self.mut_rows[wt_name] = self.mut_rows[wt_name].sample(frac=float(cfg.reduce), replace=False)

            self.wt_seqs[wt_name] = wt_rows.aa_seq[0]

    def __len__(self):
        return len(self.wt_names)

    def __getitem__(self, index):
        """Batch retrieval fxn - each batch is a single protein"""

        wt_name = self.wt_names[index]    
        mut_data = self.mut_rows[wt_name]
        wt_seq = self.wt_seqs[wt_name]

        wt_name = wt_name.split(".pdb")[0].replace("|",":")
        pdb_file = os.path.join(self.cfg.data_loc.megascale_pdbs, f"{wt_name}.pdb") # path to PDB file for this protein
        pdb = parse_pdb_cached(self.cfg, pdb_file)
        assert len(pdb[0]["seq"]) == len(wt_seq)
        pdb[0]["seq"] = wt_seq

        mutations = []
        for i, row in mut_data.iterrows():
            # no insertions, deletions, or double mutants
            if "ins" in row.mut_type or "del" in row.mut_type or ":" in row.mut_type:
                continue
            assert len(row.aa_seq) == len(wt_seq)
            wt = row.mut_type[0]
            mut = row.mut_type[-1]
            idx = int(row.mut_type[1:-1]) - 1
            assert wt_seq[idx] == wt
            assert row.aa_seq[idx] == mut

            if row.ddG_ML == '-':
                continue # filter out any unreliable data

            ddG = -torch.tensor([float(row.ddG_ML)], dtype=torch.float32)
            mutations.append(Mutation(position=idx, wildtype=wt, mutation=mut, ddG=ddG, pdb=wt_name, chain=None))

        return pdb, mutations


class FireProtDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split):

        self.cfg = cfg
        self.split = split

        filename = self.cfg.data_loc.fireprot_csv

        df = pd.read_csv(filename).dropna(subset=['ddG'])
        df = df.where(pd.notnull(df), None)

        self.seq_to_data = {}
        seq_key = "pdb_sequence"

        for wt_seq in df[seq_key].unique():
            self.seq_to_data[wt_seq] = df.query(f"{seq_key} == @wt_seq").reset_index(drop=True)

        self.df = df

        # load splits produced by mmseqs clustering
        with open(self.cfg.data_loc.fireprot_splits, 'rb') as f:
            splits = pickle.load(f)  # this is a dict with keys train/val/test and items holding FULL PDB names for a given split
            
        self.split_wt_names = {
            "val": [],
            "test": [],
            "train": [],
            "homologue-free": [],
            "all": []
        }

        self.wt_seqs = {}
        self.mut_rows = {}

        if self.split == 'all':
            all_names = list(splits.values())
            all_names = [j for sub in all_names for j in sub]
            self.split_wt_names[self.split] = all_names
        else:
            self.split_wt_names[self.split] = splits[self.split]

        self.wt_names = self.split_wt_names[self.split]

        for wt_name in self.wt_names:
            self.mut_rows[wt_name] = df.query('pdb_id_corrected == @wt_name').reset_index(drop=True)
            self.wt_seqs[wt_name] = self.mut_rows[wt_name].pdb_sequence[0]


    def __len__(self):
        return len(self.wt_names)

    def __getitem__(self, index):

        wt_name = self.wt_names[index]
        seq = self.wt_seqs[wt_name]
        data = self.seq_to_data[seq]

        pdb_file = os.path.join(self.cfg.data_loc.fireprot_pdbs, f"{data.pdb_id_corrected[0]}.pdb")
        pdb = parse_pdb_cached(self.cfg, pdb_file)

        mutations = []
        for i, row in data.iterrows():
            try:
                pdb_idx = row.pdb_position
                assert pdb[0]['seq'][pdb_idx] == row.wild_type == row.pdb_sequence[row.pdb_position]
                
            except AssertionError:  # contingency for mis-alignments
                align, *rest = pairwise2.align.globalxx(seq, pdb[0]['seq'].replace("-", "X"))
                pdb_idx = seq1_index_to_seq2_index(align, row.pdb_position)
                if pdb_idx is None:
                    continue
                assert pdb[0]['seq'][pdb_idx] == row.wild_type == row.pdb_sequence[row.pdb_position]

            ddG = None if row.ddG is None or isnan(row.ddG) else torch.tensor([row.ddG], dtype=torch.float32)
            mut = Mutation(position=pdb_idx, wildtype=pdb[0]['seq'][pdb_idx], mutation=row.mutation, ddG=ddG, pdb=wt_name, chain=None)
            mutations.append(mut)

        return pdb, mutations


class Skempi2Dataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split):

        self.cfg = cfg
        self.split = split

        filename = self.cfg.data_loc.skempi_csv

        df = pd.read_csv(filename).dropna(subset=['ddG'])
        df = df.where(pd.notnull(df), None)

        self.seq_to_data = {}
        seq_key = "pdb_sequence"

        for wt_seq in df[seq_key].unique():
            self.seq_to_data[wt_seq] = df.query(f"{seq_key} == @wt_seq").reset_index(drop=True)

        self.df = df

        # load splits produced by mmseqs clustering
        with open(self.cfg.data_loc.skempi_splits, 'rb') as f:
            splits = pickle.load(f)  # this is a dict with keys train/val/test and items holding FULL PDB names for a given split
            
        self.split_wt_names = {
            "val": [],
            "test": [],
            "train": [],
            "all": []
        }

        self.wt_seqs = {}
        self.mut_rows = {}

        if self.split == 'all':
            all_names = list(splits.values())
            all_names = [j for sub in all_names for j in sub]
            self.split_wt_names[self.split] = all_names
        else:
            self.split_wt_names[self.split] = splits[self.split]

        self.wt_names = np.array(self.split_wt_names[self.split])

        for wt_name in self.wt_names:
            mut_rows = df.query('Pdb_origin == @wt_name').reset_index(drop=True)
            if len(mut_rows) == 0:
                print(f"⚠ Skipping {wt_name}: no rows found in CSV")
                continue
            self.mut_rows[wt_name] = mut_rows
            self.wt_seqs.update({              # keys: pdbID_chainID
                f'{wt_name}_{ch_id}': seq
                for ch_id, seq in zip(
                    mut_rows['Mutation_cleaned'].str[1],
                    mut_rows['pdb_sequence']
                )
            })
        
        # Drop any wt_names that were skipped
        self.wt_names = np.array([n for n in self.wt_names if n in self.mut_rows])

    def __len__(self):
        return len(self.wt_names)

    def __getitem__(self, index):

        wt_name = self.wt_names[index]
        # seq = self.wt_seqs[wt_name]
        seqs = [seq for key, seq in self.wt_seqs.items() if key.startswith(f"{wt_name}_")]
        data = pd.concat([self.seq_to_data[seq] for seq in seqs],ignore_index=True) # combine all rows for this wt_name (PDB ID), regardless of chain

        assert wt_name == data.Pdb_origin[0]
        pdb_file = os.path.join(self.cfg.data_loc.skempi_pdbs, f"{data.Pdb_origin[0]}.pdb")
        pdb = parse_pdb_cached(self.cfg, pdb_file)
        mutations = []
        for i, row in data.iterrows():
            mut_label = row.Mutation_cleaned
            try:
                pdb_idx = (int(mut_label[2:-1])) -1 # 0-based mutant position
                wt_aa = mut_label[0]
                chain_id = mut_label[1] 
                mut_aa = mut_label[-1]
                assert pdb[0][f'seq_chain_{chain_id}'][pdb_idx] == wt_aa == row.pdb_sequence[pdb_idx]
                
            except AssertionError:  # contingency for mis-alignments
                align, *rest = pairwise2.align.globalxx(seq, pdb[0][f'seq_chain_{chain_id}'].replace("-", "X"))
                pdb_idx = seq1_index_to_seq2_index(align, row.pdb_position)
                if pdb_idx is None:
                    continue
                assert pdb[0][f'seq_chain_{chain_id}'][pdb_idx] == wt_aa == row.pdb_sequence[pdb_idx]

            ddG = None if row.ddG is None or isnan(row.ddG) else torch.tensor([row.ddG], dtype=torch.float32)
            mut = Mutation(position=pdb_idx, wildtype=wt_aa, mutation=mut_aa, ddG=ddG, pdb=wt_name, chain=chain_id)
            mutations.append(mut)

        return pdb, mutations


class ddgBenchDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, pdb_dir, csv_fname):

        self.cfg = cfg
        self.pdb_dir = pdb_dir

        df = pd.read_csv(csv_fname)
        self.df = df

        self.pdb_col = _first_existing_column(df, ["PDB", "pdb", "pdb_id", "pdb_id_corrected"]) # returns PDB column name in the benchmarking set
        self.chain_col = next((col for col in ["Chain", "chain"] if col in df.columns), None) # returns chain column name in the benchmarking set
        self.mut_col = _first_existing_column(df, ["MUT", "mutation"]) # returns mutation column name in the benchmarking set
        self.seq_col = _first_existing_column(df, ["SEQ", "wt_seq", "wt_sequence"]) # returns WT sequence column name in the benchmarking set
        self.target_col = next((col for col in ["DDG", "ddG", "dTm", "DDG_ML", "ddG_ML"] if col in df.columns), None) # returns prediction target column name in the benchmarking set
        if self.target_col is None:
            raise KeyError(f"No target column found in benchmark CSV: {list(df.columns)}")
        self.target_sign = -1.0 if "ddg" in self.target_col.lower() else 1.0

        self.wt_seqs = {}
        self.mut_rows = {}
        self.chains = {}
        self.wt_names = [str(wt_name).replace(".pdb", "") for wt_name in df[self.pdb_col].dropna().astype(str).unique()]

        for wt_name in self.wt_names:
            mut_rows = df.loc[df[self.pdb_col].astype(str).str.replace(".pdb", "", regex=False) == wt_name].reset_index(drop=True)
            if len(mut_rows) == 0:
                continue

            self.mut_rows[wt_name] = mut_rows
            # self.chains[wt_name] = _normalize_chain_value(mut_rows[self.chain_col].iloc[0]) if self.chain_col is not None else None
            raw_vals = mut_rows[self.chain_col].dropna().astype(str).unique()
            chains_set = []
            for v in raw_vals:
                norm = _normalize_chain_value(v)
                if norm is None:
                    continue
                if isinstance(norm, list):
                    chains_set.extend(norm)
                else:
                    chains_set.append(norm)

            # de-duplicate while preserving order
            seen = set()
            chains = [c for c in chains_set if not (c in seen or seen.add(c))]
            self.chains[wt_name] = chains if len(chains) > 0 else None

            if 'S669' not in self.pdb_dir:
                self.wt_seqs[wt_name] = _clean_benchmark_sequence(mut_rows[self.seq_col].iloc[0])

    def __len__(self):
        return len(self.wt_names)

    def __getitem__(self, index):
        """Batch retrieval fxn - each batch is a single protein"""

        wt_name = self.wt_names[index]
        chain = self.chains.get(wt_name)
        mut_data = self.mut_rows[wt_name]
        # wt_seq = self.wt_seqs.get(wt_name)

        pdb_file = os.path.join(self.pdb_dir, wt_name + '.pdb')

        pdb = alt_parse_PDB(pdb_file, chain)
        wt_seq = pdb[0]['seq']
        resn_list = pdb[0]['resn_list']
        # print(f'[DEBUG] {wt_seq}')

        # if wt_seq is None:
        #     wt_seq = _clean_benchmark_sequence(mut_data[self.seq_col].iloc[0])
        pdb_seq = pdb[0]["seq"]
        align = None
        # if wt_seq != pdb_seq:
        #     align = pairwise2.align.globalxx(wt_seq, pdb_seq.replace("-", "X"))[0]
            # print(f"Warning: WT sequence and PDB sequence do not match for {wt_name}. Alignment will be used to map mutation positions. Consider updating the benchmark CSV with cleaned sequences that match the PDBs to avoid this step. WT seq: {wt_seq}, PDB seq: {pdb_seq}, Alignment: {align}")

        mutations = []
        for i, row in mut_data.iterrows():
            mut_info = row[self.mut_col]
            wtAA, pos, mutAA = _parse_mutation_label(mut_info)
            
            try:
                pdb_idx = resn_list.index(str(pos)) # index of the mutant position in the sequence
            except ValueError:
                print(f"Position {pos} not found in PDB sequence for {wt_name}")
                continue
            
            # print(f'[DEBUG]: CSV sequence index = {pdb_idx}')
            if pdb_idx < 0 or pdb_idx >= len(wt_seq):
                print(f"Position {pos} out of bounds for WT sequence length {len(wt_seq)} for {wt_name}")
                continue
            if wt_seq[pdb_idx] != wtAA:
                # print(f'[DEBUG]:{resn_list}')
                # print(f'pdb_idx: {pdb_idx}')
                # print(f'wt_seq: {wt_seq}')
                # print(f'pdb_seq: {pdb_seq}')
                print(f"WT AA mismatch at position {pos} for {wt_name}: expected {wtAA}, found {wt_seq[pdb_idx]}")
                continue

            # try:
            #     if pdb_idx >= len(pdb_seq) or pdb_seq[pdb_idx] != wtAA:
            #         continue
            #     if pdb_idx is None or pdb_idx >= len(pdb_seq):
            #         continue
            # except Exception:
            #     continue

            target_value = row[self.target_col]
            ddG = None if pd.isna(target_value) else torch.tensor([float(target_value) * self.target_sign], dtype=torch.float32)
            mut = Mutation(pdb_idx, wtAA, mutAA, ddG, wt_name)
            mutations.append(mut)

        return pdb, mutations 


class ComboDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split):

        datasets = []
        if "fireprot" in cfg.datasets:
            fireprot = FireProtDataset(cfg, split)
            datasets.append(fireprot)
        if "megascale" in cfg.datasets:
            mega_scale = MegaScaleDataset(cfg, split)
            datasets.append(mega_scale)
        self.mut_dataset = ConcatDataset(datasets)

    def __len__(self):
        return len(self.mut_dataset)

    def __getitem__(self, index):
        return self.mut_dataset[index]


