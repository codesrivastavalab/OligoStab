# OligoStab

OligoStab integrates structural and sequential features to predict changes in thermostability due to single-point mutations. The model is adapted to oligomeric proteins using a positional jump modification in amino acid sequence processing and a fine-tuning strategy. 


## Installation
To install OligoStab on your local system clone this repository:
```
git clone https://github.com/codesrivastavalab/OligoStab.git
cd OligoStab
```
Create and activate environemnt:
```
conda env create -f requirements.yaml
conda activate oligostab
```

## Inference
Run inside `ddg_pipeline` directory.
Use the monomer model checkpoint for single chain proteins and oligomer checkpoint for protein complexes. 
### Single PDB with site-saturation mutagenesis
```
python test_model.py \
--checkpoint path-to-model-checkpoint.ckpt \
--pdb input-PDB.pdb \
--all-mutations 
--config configs/config.yaml \
--output pred-results/file-name.csv
```

### Single PDB with specified mutation list in a CSV
The 
```
python test_model.py \
--checkpoint path-to-model-checkpoint.ckpt \
--pdb input-PDB.pdb \
--csv mutation-list.csv \
--mutation-column mutation \
--config configs/config.yaml \
--output pred-results/file-name.csv
```

### Multiple PDBs
```
python test_model.py 
--checkpoint path-to-model-checkpoint.ckpt \
--csv mutation-list.csv \
--pdb-column pdb \
--mutation-column mutation \
--config configs/config.yaml \
--output pred-results/file-name.csv
```

The output CSV from `test_model.py` includes the following columns:
- pdb
- mutation_str
- predicted_ddG

## Training


## Benchmarking


## Dataset Availability 

