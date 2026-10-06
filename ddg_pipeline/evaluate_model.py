"""Evaluate a trained ddG stability model on the test split (FireProt or MegaScale).

Usage example:
  python evaluate_model.py --checkpoint checkpoints/my_model.ckpt --config ddg_pipeline/configs/config.yaml --out results.csv
"""
import argparse
import sys
import torch
import numpy as np
import csv
from pathlib import Path

# Add ddg_pipeline directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from datasets import FireProtDataset, MegaScaleDataset, ddgBenchDataset, Skempi2Dataset, ComboDataset

from omegaconf import OmegaConf
from torchmetrics import R2Score, MeanSquaredError, SpearmanCorrCoef, PearsonCorrCoef, MeanAbsoluteError
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

def load_config(args):
    yaml_path = args.config if args.config is not None else "configs/config.yaml"
    cfg = OmegaConf.load(yaml_path)
    # Merge local.yaml if present
    try:
        cfg = OmegaConf.merge(cfg, OmegaConf.load("configs/local.yaml"))
    except FileNotFoundError:
        pass
    # Merge CLI overrides
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

    # Allow overriding dataset via CLI argument (except special values handled in main)
    if args.dataset is not None and args.dataset not in ("all", "both"):
        cfg.datasets = [args.dataset]

    return cfg


def get_model_class(mode):
    if mode == "classification":
        from train_ddg_model_classification import CompiledModelPL as ModelClass
    else:
        # from train_ddg_model_regression import CompiledModelPL as ModelClass
        from ddg_pipeline.train_ddG_model import CompiledModelPL as ModelClass
    return ModelClass


def build_dataset(cfg, dataset_name):
    if dataset_name == 'fireprot':
        return FireProtDataset(cfg, 'homologue-free')
    if dataset_name == 'megascale_s669':
        return MegaScaleDataset(cfg, 'test')
    if dataset_name.startswith('megascale_cv'):
        cv = dataset_name[-1]
        return MegaScaleDataset(cfg, f'cv_test_{cv}')
    if dataset_name == 'megascale':
        return MegaScaleDataset(cfg, 'test')
    if dataset_name == 'skempi2':
        return Skempi2Dataset(cfg, 'test')
    if dataset_name == 'combo':
        from datasets import ComboDataset
        return ComboDataset(cfg, 'test')
    raise ValueError("Invalid dataset specified in config or via --dataset")


def resolve_datasets(cfg, dataset_arg):
    if dataset_arg in ("all", "both"):
        return ["fireprot", "megascale"]
    if dataset_arg is not None:
        return [dataset_arg]
    if len(cfg.datasets) == 1:
        return [cfg.datasets[0]]
    return ["combo"]


def evaluate_dataset(model_pl, cfg, dataset_name, out_path, mode, device):
    ds = build_dataset(cfg, dataset_name)

    use_normalization = bool(cfg.training.get('use_normalization', True))

    out_rows = []
    reg_pred_ddg = []
    reg_true_ddg = []
    cls_pred = []
    cls_true = []

    n = len(ds)
    for i in range(n):
        pdb, mutations = ds[i]
        if len(mutations) == 0:
            continue

        # model expects list of Mutation objects; ensure dataset emits labeled examples
        cfg.training.train_flag = True

        with torch.no_grad():
            fwd_preds, rev_preds = model_pl(
                pdb,
                mutations,
                pdb_type='monomer' if dataset_name == 'fireprot' or dataset_name == 'megascale' else 'heteromer',
                num_chains=1,
            )

        preds_norm_t = fwd_preds.detach()
        if use_normalization and hasattr(model_pl, "denormalize_ddg"):
            preds_denorm_t = model_pl.denormalize_ddg(preds_norm_t)
        else:
            preds_denorm_t = preds_norm_t

        preds_norm = preds_norm_t.cpu().numpy() if isinstance(preds_norm_t, torch.Tensor) else np.asarray(preds_norm_t)
        preds_denorm = preds_denorm_t.cpu().numpy() if isinstance(preds_denorm_t, torch.Tensor) else np.asarray(preds_denorm_t)

        for j, mut in enumerate(mutations):
            true_val = None
            if mut.ddG is not None:
                if isinstance(mut.ddG, torch.Tensor):
                    true_val = float(mut.ddG.detach().cpu().item())
                else:
                    true_val = float(mut.ddG)

            out_rows.append(
                (
                    i,
                    pdb[0].get('name', ''),
                    f'{mut.wildtype}{mut.position}{mut.mutation}',
                    float(preds_norm[j]),
                    float(preds_denorm[j]),
                    true_val,
                )
            )

            if true_val is not None:
                # Metrics should be computed on original ddG scale.
                reg_pred_ddg.append(float(preds_denorm[j]))
                reg_true_ddg.append(true_val)

    # write per-mutation predictions
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        if mode == "classification":
            w.writerow(['protein_idx', 'pdb_name', 'mutation', 'pred_class', 'true_class'])
        else:
            w.writerow([
                'protein_idx',
                'pdb_name',
                'mutation',
                'pred_ddG_normalized',
                'pred_ddG_denormalized',
                'true_ddG',
            ])
        for r in out_rows:
            w.writerow(r)

    metrics_path = out_path.with_suffix(out_path.suffix + '.metrics.csv')

    if mode == "classification":
        if len(cls_pred) == 0:
            return

        y_true = np.array(cls_true)
        y_pred = np.array(cls_pred)

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        rec = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        mcc = float(matthews_corrcoef(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])

        # Stable is the positive class.
        y_true_pos = (y_true == -1).astype(int)
        y_pred_pos = (y_pred == -1).astype(int)
        stable_precision = float(precision_score(y_true_pos, y_pred_pos, zero_division=0))
        stable_recall = float(recall_score(y_true_pos, y_pred_pos, zero_division=0))
        stable_f1 = float(f1_score(y_true_pos, y_pred_pos, zero_division=0))

        print(f"Dataset: {dataset_name}")
        print(f"Accuracy: {acc:.3f}")
        print(f"Precision (stable): {stable_precision:.3f}")
        print(f"Recall (stable): {stable_recall:.3f}")
        print(f"F1 (stable): {stable_f1:.3f}")
        # print(f"Precision (macro): {prec:.3f}")
        # print(f"Recall (macro): {rec:.3f}")
        # print(f"F1 (macro): {f1:.3f}")
        print(f"MCC: {mcc:.3f}")
        print("Confusion matrix rows=true, cols=pred, labels=[-1,0,1]:")
        print(cm)

        with open(metrics_path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['metric', 'value'])
            w.writerow(['accuracy', acc])
            w.writerow(['precision_macro', prec])
            w.writerow(['recall_macro', rec])
            w.writerow(['f1_macro', f1])
            w.writerow(['mcc', mcc])
            w.writerow(['precision_stable', stable_precision])
            w.writerow(['recall_stable', stable_recall])
            w.writerow(['f1_stable', stable_f1])
            w.writerow(['cm_label_order', '[-1, 0, 1]'])
            for i in range(3):
                for j in range(3):
                    w.writerow([f'cm_{i}_{j}', int(cm[i, j])])

    else:
        if len(reg_pred_ddg) == 0:
            return

        preds = np.array(reg_pred_ddg)
        trues = np.array(reg_true_ddg)

        # compute metrics using torchmetrics
        preds_t = torch.tensor(preds, dtype=torch.float32, device=device)
        trues_t = torch.tensor(trues, dtype=torch.float32, device=device)

        spearman_m = SpearmanCorrCoef().to(device)
        pearson_m = PearsonCorrCoef().to(device)
        rmse_m = MeanSquaredError(squared=False).to(device)
        r2_m = R2Score().to(device)
        mae_m = MeanAbsoluteError().to(device)

        def _safe(metric, a, b):
            try:
                return float(metric(a, b).detach().cpu().numpy())
            except Exception:
                return float('nan')

        spearman = _safe(spearman_m, preds_t, trues_t)
        pearson = _safe(pearson_m, preds_t, trues_t)
        rmse = _safe(rmse_m, preds_t, trues_t)
        r2 = _safe(r2_m, preds_t, trues_t)
        mae = _safe(mae_m, preds_t, trues_t)

        print(f"Dataset: {dataset_name}")
        print(f"Spearman: {spearman:.3f}")
        print(f"Pearson: {pearson:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2: {r2:.3f}")
        print(f"MAE: {mae:.3f}")

        with open(metrics_path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['metric', 'value'])
            w.writerow(['spearman', spearman])
            w.writerow(['pearson', pearson])
            w.writerow(['rmse', rmse])
            w.writerow(['r2', r2])
            w.writerow(['mae', mae])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', help='Path to model/config YAML')
    p.add_argument('--dataset', help='Override dataset name (e.g., fireprot, megascale, megascale_cv0, megascale_s669, both)')
    p.add_argument('--mode', choices=['regression', 'classification'], default='regression', help='Evaluation mode')
    p.add_argument('--out', default='test-results/TEST-fire.csv')
    args = p.parse_args()

    cfg = load_config(args)

    # load trained Lightning module from checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ModelClass = get_model_class(args.mode)
    # Load permissively so older checkpoints missing ddg_min/ddg_max buffers still work.
    strict_load = False
    model_pl = ModelClass.load_from_checkpoint(
        args.checkpoint,
        config=cfg,
        strict=strict_load,
    )
    model_pl = model_pl.to(device)
    model_pl.eval()

    datasets_to_run = resolve_datasets(cfg, args.dataset)
    base_out = Path(args.out)

    for ds_name in datasets_to_run:
        if len(datasets_to_run) > 1:
            out_path = base_out.with_name(f"{base_out.stem}_{ds_name}{base_out.suffix}")
        else:
            out_path = base_out

        evaluate_dataset(
            model_pl=model_pl,
            cfg=cfg,
            dataset_name=ds_name,
            out_path=out_path,
            mode=args.mode,
            device=device,
        )


if __name__ == '__main__':
    main()