# OligoStab

OligoStab integrates structural and sequential features to predict changes in thermostability due to single-point mutations. The model is adapted to oligomeric proteins using a positional jump modification in amino acid sequence processing and a fine-tuning strategy. 


## Installation
To install OligoStab on your local system clone this repository:
```
git clone https://github.com/codesrivastavalab/OligoStab.git
cd OligoStab
```
Create and activate environment (if you are using Apple silicon chip, use `requirements-mac.yaml`):
```
conda env create -f requirements.yaml
conda activate oligostab
```
The pre-trained monomer and oligomer fine-tuned weights for OligoStab can be download from Hugging Face: [OligoStab weights](https://huggingface.co/Raghav0573/oligostab-weights/tree/main)
- Monomer weights: oligostab-monomer.ckpt
- Oligomer weights: oligostab-FT.ckpt

The following steps can also be used to download weights directly from your terminal:
1. Install Hugging Face Hub in your terminal, if not already present
```
pip install "huggingface_hub>=0.34,<1.0"
```
2. Download the weights
```
hf download Raghav0573/oligostab-weights oligostab-FT.ckpt --local-dir ./weights
```

Datasets used for training, testing and benchmarking can be accessed from hugging face: [Datasets](https://huggingface.co/datasets/Raghav0573/oligostab-data/tree/main) 

To download the dataset use the following command:
```
hf download Raghav0573/oligostab-data \
--repo-type dataset \
--include "<dataset_name>/**" \       
--local-dir ./data
```

## Google Colab implmentation
For a quick, user-friendly way to generate predictions using OligoStab without any installations use this [Colab link](https://colab.research.google.com/drive/11nfgqQWshSq_QC8XewXFdApO5T25A3Wu?usp=sharing).

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
 
```
python test_model.py \
--checkpoint ../weights/oligostab-FT.ckpt \ 
--pdb ../example/3vub_dimer_mod.pdb \
--csv ../example/RV_data_muts.csv \ 
--mutation-column mutation \
--config configs/config.yaml \
--output pred-results/CcdB-test.csv
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
- pdb - PDB ID (eg. 3VUB)
- position - Residue position in 0-index
- wildtype - Wildtype amino acid
- mutation - Mutant amino acid
- mutation_str - Mutation label (eg. R2M)
- predicted_ddG - Predicted $\Delta \Delta G$ in kcal/mol

## Training


## Benchmarking


## Dataset Availability 

