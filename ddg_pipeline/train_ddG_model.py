import sys
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from omegaconf import OmegaConf

from compiled_model_multimer_ablation import StabilityPredictionModel
from datasets import FireProtDataset, MegaScaleDataset, Skempi2Dataset, ComboDataset
from transformers import get_cosine_schedule_with_warmup
import bitsandbytes as bnb

import time


# def compute_ddg_normalization_stats(train_dataset):
#     """Compute min/max of ddG values from training split.
    
#     Returns:
#         ddg_min: Minimum ddG value in training set
#         ddg_max: Maximum ddG value in training set
#     """
#     all_ddg = []
    
#     for i in range(len(train_dataset)):
#         _, mutations = train_dataset[i]
#         for mut in mutations:
#             if mut.ddG is not None:
#                 if isinstance(mut.ddG, torch.Tensor):
#                     ddg_val = mut.ddG.detach().cpu().float().item()
#                 else:
#                     ddg_val = float(mut.ddG)
#                 all_ddg.append(ddg_val)
    
#     if len(all_ddg) == 0:
#         raise ValueError("No ddG values found in training dataset")
    
#     ddg_min = float(min(all_ddg))
#     ddg_max = float(max(all_ddg))
    
#     print(f"✓ Normalization stats computed: ddG range [{ddg_min:.4f}, {ddg_max:.4f}]")
#     return ddg_min, ddg_max


def get_metrics():
    return {
        "r2": R2Score(),
        "mse": MeanSquaredError(squared=True),
        "rmse": MeanSquaredError(squared=False),
        "spearman": SpearmanCorrCoef(),
        "pearson": PearsonCorrCoef()
    }

def compute_magnitude_aware_weights(ddg_values, n_bins=20, precision_bias=1.5):
    """
    Inverse-frequency weights with an extra bias for large-magnitude
    stabilising mutations. This downweights near-zero/noisy cases and
    emphasises physically meaningful, high-confidence examples.
    """
    if torch.is_tensor(ddg_values):
        values = ddg_values.detach().cpu().float().numpy()
    else:
        values = np.asarray(ddg_values, dtype=np.float32)

    if values.ndim > 1:
        values = values.reshape(-1)

    if values.size == 0:
        return torch.zeros(0, dtype=torch.float32)

    counts, bin_edges = np.histogram(values, bins=n_bins)
    bin_indices = np.digitize(values, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Base inverse-frequency weight
    freq = counts / counts.sum()
    weights = 1.0 / (freq[bin_indices] + 1e-6)

    # Additional magnitude bias for strongly stabilising mutations
    magnitude_bias = np.where(
        values < 0,
        1.0 + precision_bias * np.abs(values) / (np.abs(values).max() + 1e-6),
        1.0
    )
    weights = weights * magnitude_bias

    # Normalise so average weight = 1 (keeps loss scale stable)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

class CompiledModelPL(pl.LightningModule):
    ''' PyTorch Lightning module for normalized regression ddG prediction.

    Training strategy:
    - Freeze ProteinMPNN and ESM-2 weights (pre-trained backbones).
    - Trains only CrossAttentionLayer and MLP weights.
    - Normalizes targets to [-1, 1] using min-max normalization.
    - Uses MSE loss between normalized predictions and normalized targets.
    - Denormalizes for metric computation and inference.
    - Filters out samples with missing experimental ddG values.
    '''
    def __init__(self, config, ddg_min=None, ddg_max=None):
        super().__init__()
        self.config = config
        self.model = StabilityPredictionModel(config)
        self.lr = config.training.learn_rate
        self.lambda_sym = 1.5  # weight for symmetry loss
        
        # # Normalization parameters: map [ddg_min, ddg_max] -> [-1, 1]
        # self.register_buffer("ddg_min_buffer", torch.tensor(ddg_min if ddg_min is not None else 0.0, dtype=torch.float32))
        # self.register_buffer("ddg_max_buffer", torch.tensor(ddg_max if ddg_max is not None else 1.0, dtype=torch.float32))
        # self.ddg_min = ddg_min if ddg_min is not None else 0.0
        # self.ddg_max = ddg_max if ddg_max is not None else 1.0

        # Freeze ProteinMPNN and ESM-2 weights
        self._freeze_backbones() # runs this method while initializing

        # Loss function (for normalized space)
        self.mse_loss = nn.MSELoss(reduction='mean')

        # set up metrics dictionary (will use denormalized values)
        self.metrics = nn.ModuleDict()
        for split in ("train_metrics", "val_metrics"):
            self.metrics[split] = nn.ModuleDict()
            out = "ddG"
            self.metrics[split][out] = nn.ModuleDict()
            for name, metric in get_metrics().items():
                self.metrics[split][out][name] = metric

    def _freeze_backbones(self):
        """Freeze ProteinMPNN and ESM-2 weights; keep CrossAttention and MLP trainable."""
        finetune_enabled = bool(getattr(self.config.finetuning, "finetune", False))
        finetune_mode = str(getattr(self.config.finetuning, "mode", "mlp_only")).lower()

        # Freeze ProteinMPNN
        for param in self.model.mpnn_model.parameters():
            param.requires_grad = True if not self.config.model.freeze_weights else False

        # Freeze ESM-2
        for param in self.model.esm_model.parameters():
            param.requires_grad = False

        # Ensure cross-attention and MLP are trainable
        for param in self.model.cross_attention.parameters():
            param.requires_grad = True
        for param in self.model.mpnn_score_proj.parameters():
            param.requires_grad = True
        for param in self.model.mlp.parameters():
            param.requires_grad = True

        # During finetuning, enforce explicit head-update modes.
        if finetune_enabled:
            if finetune_mode == "mlp_only":
                print("✓ Finetuning mode=mlp_only: training only MLP.")
                for param in self.model.mpnn_model.parameters():
                    param.requires_grad = False
                for param in self.model.esm_model.parameters():
                    param.requires_grad = False
                for param in self.model.cross_attention.parameters():
                    param.requires_grad = False
                for param in self.model.mpnn_score_proj.parameters():
                    param.requires_grad = False
                for param in self.model.mlp.parameters():
                    param.requires_grad = True

            elif finetune_mode == "lora_mlp":
                print("✓ Finetuning mode=lora_mlp: training LoRA adapters in cross-attention + MLP.")
                if not getattr(self.model.cross_attention, "use_lora", False):
                    raise ValueError(
                        "finetuning.mode='lora_mlp' requires LoRA-enabled cross-attention. "
                        "Set finetuning.use_lora_cross_attention=True and finetuning.lora_rank>0."
                    )

                for param in self.model.mpnn_model.parameters():
                    param.requires_grad = False
                for param in self.model.esm_model.parameters():
                    param.requires_grad = False

                # Freeze full cross-attention first; re-enable only LoRA adapter matrices.
                for param in self.model.cross_attention.parameters():
                    param.requires_grad = False
                for name, param in self.model.cross_attention.named_parameters():
                    if "lora_A" in name or "lora_B" in name:
                        param.requires_grad = True

                for param in self.model.mpnn_score_proj.parameters():
                    param.requires_grad = False
                for param in self.model.mlp.parameters():
                    param.requires_grad = True

            else:
                raise ValueError(
                    f"Invalid finetuning.mode='{finetune_mode}'. Expected one of ['mlp_only', 'lora_mlp']."
                )

        print("✓ ProteinMPNN and ESM-2 weights frozen.")
        print(f"✓ Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"✓ Frozen params: {sum(p.numel() for p in self.model.parameters() if not p.requires_grad):,}")

    # def normalize_ddg(self, ddg):
    #     """Normalize ddG values to [-1, 1].
        
    #     Args:
    #         ddg: Tensor or scalar of original ddG values
        
    #     Returns:
    #         Normalized values in [-1, 1]
    #     """
    #     # normalized = 2 * (x - min) / (max - min) - 1
    #     if isinstance(ddg, torch.Tensor):
    #         ddg_min = self.ddg_min_buffer if hasattr(self, 'ddg_min_buffer') else torch.tensor(self.ddg_min, device=ddg.device)
    #         ddg_max = self.ddg_max_buffer if hasattr(self, 'ddg_max_buffer') else torch.tensor(self.ddg_max, device=ddg.device)
    #     else:
    #         ddg_min = self.ddg_min
    #         ddg_max = self.ddg_max
        
    #     normalized = 2.0 * (ddg - ddg_min) / (ddg_max - ddg_min) - 1.0
    #     return normalized
    
    # def denormalize_ddg(self, normalized_ddg):
    #     """Denormalize from [-1, 1] back to original scale.
        
    #     Args:
    #         normalized_ddg: Tensor or scalar of normalized values in [-1, 1]
        
    #     Returns:
    #         Original-scale ddG values
    #     """
    #     # denormalized = (normalized + 1) / 2 * (max - min) + min
    #     if isinstance(normalized_ddg, torch.Tensor):
    #         ddg_min = self.ddg_min_buffer if hasattr(self, 'ddg_min_buffer') else torch.tensor(self.ddg_min, device=normalized_ddg.device)
    #         ddg_max = self.ddg_max_buffer if hasattr(self, 'ddg_max_buffer') else torch.tensor(self.ddg_max, device=normalized_ddg.device)
    #     else:
    #         ddg_min = self.ddg_min
    #         ddg_max = self.ddg_max
        
    #     denormalized = (normalized_ddg + 1.0) / 2.0 * (ddg_max - ddg_min) + ddg_min
    #     return denormalized

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        """Configure optimizer for trainable parameters (cross-attention + MLP)."""
        # Get only trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # AdamW optimizer with reasonable defaults
        # optimizer = torch.optim.AdamW(
        #     trainable_params,
        #     lr=self.lr/10 if self.stage==2 else self.lr,  # decrease LR for fine-tuning
        #     weight_decay=1e-3, # L2 regularization
        #     betas=(0.9, 0.999),
        #     eps=1e-8)
        optimizer = bnb.optim.AdamW8bit(
            trainable_params,
            lr=self.lr/10 if self.stage==2 else self.lr,  # decrease LR for fine-tuning
            weight_decay=1e-3, # L2 regularization
            betas=(0.9, 0.999),
            eps=1e-8)

        # Learning rate scheduler
        # Cosine annealing with warmup works well for fine-tuning
        epochs = self.config.training.max_epochs
        total_steps = 2*epochs
        warmup_steps = int(0.1 * total_steps) # 10% Warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=epochs)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.config.training.max_epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def shared_eval(self, batch, batch_idx, prefix):
        """
        Shared evaluation logic for training, validation, and testing.
        Can use normalized or non-normalized training depending on config.

        Training modes:
        - Normalized (default): Normalizes targets to [-1, 1], computes loss on normalized space
        - Non-normalized: Uses original ddG values for loss and metrics
        
        To switch modes, set config.training.use_normalization = False (or True)

        Args:
            batch: tuple of (mut_pdb, mutations) where mutations is a list of mutation objects
                   each with optional .ddG attribute
            batch_idx: batch index
            prefix: 'train', 'val', or 'test' for logging

        Returns:
            loss (tensor or None if no valid samples)
        """
        # Get normalization flag from config (default: True for backward compatibility)
        # use_normalization = self.config.training.get('use_normalization', True)
        
        assert len(batch) == 1, f"Expected batch size 1, got {len(batch)}"
        pdb, mutations = batch[0]

        # Forward pass: predict ddG for all mutants
        pred_ddg_fwd, pred_ddg_rev = self(pdb, mutations)  # shape: [num_mutants]
        true_ddg = []

        # Collect losses only for mutations with experimental ddG available
        for mut in mutations:
            if mut.ddG is not None:
                true_ddg.append(mut.ddG)

        if len(pred_ddg_fwd) == 0:
            return None

        # Extract predictions and true values for valid samples
        pred_tensor = pred_ddg_fwd[[i for i, mut in enumerate(mutations) if mut.ddG is not None]]
        true_tensor = torch.stack(true_ddg).squeeze(-1)   # shape [K] in original scale
        
        # Choose training protocol based on config flag
        # if use_normalization:
        #     # NORMALIZED PROTOCOL: Use [-1, 1] normalized space
        #     pred_tensor_for_loss = self.normalize_ddg(pred_tensor)
        #     true_tensor_for_loss = self.normalize_ddg(true_tensor)
        #     pred_tensor_for_metrics = self.denormalize_ddg(pred_tensor_for_loss)
        #     true_tensor_for_metrics = true_tensor
        # else:
            # NON-NORMALIZED PROTOCOL: Use original scale directly
        pred_tensor_for_loss = pred_tensor
        true_tensor_for_loss = true_tensor
        pred_tensor_for_metrics = pred_tensor
        true_tensor_for_metrics = true_tensor

        # Compute magnitude-aware weighted regression loss for training only
        if prefix == 'train' and true_tensor.numel() > 0:
            sample_weights = compute_magnitude_aware_weights(true_tensor)
            sample_weights = sample_weights.to(device=pred_tensor_for_loss.device, dtype=pred_tensor_for_loss.dtype)
            loss_mse = torch.mean(sample_weights * (pred_tensor_for_loss - true_tensor_for_loss)**2)
            loss_sym = F.mse_loss(pred_ddg_fwd, -pred_ddg_rev)
            total_loss = loss_mse + self.lambda_sym * loss_sym
        else:
            loss_mse = F.mse_loss(pred_tensor_for_loss, true_tensor_for_loss)
            loss_sym = F.mse_loss(pred_ddg_fwd, -pred_ddg_rev)  # symmetry loss for all mutants
            total_loss = loss_mse + self.lambda_sym * loss_sym
        
        # Update metrics with appropriate scale values
        for metric in self.metrics[f"{prefix}_metrics"]["ddG"].values():
            metric.update(pred_tensor_for_metrics, true_tensor_for_metrics)

        # Determine logging frequency
        on_step = False
        on_epoch = not on_step

        # Log metrics
        output = "ddG"
        for name, metric in self.metrics[f"{prefix}_metrics"][output].items():
            self.log(f"{prefix}_{output}_{name}", metric, prog_bar=True,
                    on_step=on_step, on_epoch=on_epoch, batch_size=len(batch))

        # Log loss
        self.log(f"{prefix}_loss_mse", loss_mse, prog_bar=True, on_step=on_step, on_epoch=on_epoch, batch_size=len(batch))
        self.log(f"{prefix}_loss_sym", loss_sym, prog_bar=True, on_step=on_step, on_epoch=on_epoch, batch_size=len(batch))
        self.log(f"{prefix}_loss_total", total_loss, prog_bar=True, on_step=on_step, on_epoch=on_epoch, batch_size=len(batch))

        # Return None if no valid samples, otherwise return loss
        if total_loss == 0.0:
            return None
        return total_loss

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'test')

def train(config):
    print('Configuration:\n', config)

    if 'project' in config:
        wandb.init(project=config.project, name=config.name)
    else:
        config.name = 'test'

    # load the specified dataset
    if len(config.datasets) == 1: # one dataset training
        dataset = config.datasets[0]
        if dataset == 'fireprot':
            train_dataset = FireProtDataset(config, "train")
            val_dataset = FireProtDataset(config, "val")
        elif dataset == 'megascale_s669':
            train_dataset = MegaScaleDataset(config, "train_s669")
            val_dataset = MegaScaleDataset(config, "val")
        elif dataset.startswith('megascale_cv'):
                cv = dataset[-1]
                train_dataset = MegaScaleDataset(config, f"cv_train_{cv}")
                val_dataset = MegaScaleDataset(config, f"cv_val_{cv}")
        elif dataset == 'megascale':
                train_dataset = MegaScaleDataset(config, "train")
                val_dataset = MegaScaleDataset(config, "val")
        # elif dataset == 'skempi2':
        #     train_dataset = Skempi2Dataset(config, "train")
        #     val_dataset   = Skempi2Dataset(config, "val")
        else:
            raise ValueError("Invalid dataset specified!")
    else:
        train_dataset = ComboDataset(config, "train")
        val_dataset = ComboDataset(config, "val")
    
    if config.finetuning.finetune:
        train_dataset = Skempi2Dataset(config, "train")
        val_dataset   = Skempi2Dataset(config, "val")
    
    # ddg_min, ddg_max = compute_ddg_normalization_stats(train_dataset)  # Compute normalization statistics from training set

    if 'num_workers' in config.training:
        train_workers, val_workers = int(config.training.num_workers * 0.75), int(config.training.num_workers * 0.25)
    else:
        train_workers, val_workers = 0, 0

    train_loader = DataLoader(train_dataset, collate_fn=lambda x: x, shuffle=True, num_workers=train_workers)
    val_loader = DataLoader(val_dataset, collate_fn=lambda x: x, num_workers=val_workers)

    # Initialize model with normalization parameters
    model_pl = CompiledModelPL(config)
    model_pl.stage = 1  # initial training stage
    pretrained_ckpt = getattr(config.finetuning, "pretrained_checkpoint", None)
    if config.finetuning.finetune and pretrained_ckpt:
        ckpt = torch.load(pretrained_ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        missing, unexpected = model_pl.model.load_state_dict(state_dict, strict=False)
        model_pl.lr = config.finetuning.FT_learn_rate
        if missing:
            print(f"⚠ Missing keys: {missing[:5]}")
        print(f"✓ Loaded pretrained weights from {pretrained_ckpt}")

    filename = config.name + '_{epoch:02d}_{val_ddG_spearman:.02}'
    checkpoint_callback = ModelCheckpoint(monitor='val_ddG_spearman', mode='max', dirpath='checkpoints', filename=filename)
    early_stop_callback = EarlyStopping(monitor='val_ddG_mse', min_delta=0.00, patience=10, verbose=False, mode='min')
    logger = WandbLogger(project=config.project, name="Expt-1-ES", log_model=False) if 'project' in config else None
    max_ep = config.training.max_epochs if 'max_epochs' in config.training else 100

    trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stop_callback], logger=logger, log_every_n_steps=10, max_epochs=max_ep,
                         accelerator=config.platform.accel, devices=1)
    t0 = time.time()
    trainer.fit(model_pl, train_loader, val_loader)
    tot_time = time.time()-t0
    print(f"Total training time: {tot_time:.1f} s")

    if 'two_stage' in config.training:  # sequential fine-tuning
        if config.training.two_stage:
            print('Two-stage Training Enabled')
            del trainer, train_dataset, val_dataset, train_loader, val_loader
            # load new datasets for further training
            train_dataset = FireProtDataset(config, "train")
            val_dataset = FireProtDataset(config, "val")
            train_loader = DataLoader(train_dataset, collate_fn=lambda x: x, shuffle=True, num_workers=train_workers)
            val_loader = DataLoader(val_dataset, collate_fn=lambda x: x, num_workers=val_workers)

            model_pl.stage = 2
            # Optionally freeze cross-attention for finetuning and train only MLP
            if str(getattr(config.finetuning, "mode", "mlp_only")).lower() == "mlp_only":
                print('Finetune mode: freezing cross-attention, training MLP only')
                # Freeze cross-attention parameters
                for param in model_pl.model.cross_attention.parameters():
                    param.requires_grad = False
                # Ensure MLP parameters are trainable
                for param in model_pl.model.mlp.parameters():
                    param.requires_grad = True
            # re-start training with a new trainer
            trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stop_callback], logger=logger, log_every_n_steps=10, max_epochs=max_ep,
                                accelerator=config.platform.accel, devices=1)
            trainer.fit(model_pl, train_loader, val_loader, ckpt_path=checkpoint_callback.best_model_path)

if __name__ == "__main__":
    # config.yaml and local.yaml files are combined to assemble all runtime arguments
    if len(sys.argv) == 1:
        yaml = "ddg_pipeline/configs/config.yaml"
    else:
        yaml = sys.argv[1]

    cfg = OmegaConf.load(yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.load("ddg_pipeline/configs/local.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    train(cfg)
