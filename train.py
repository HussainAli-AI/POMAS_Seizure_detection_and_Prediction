"""Training loop for SALT LTformer detection model.

Supports:
- Mixed precision training (6GB VRAM constraint)
- Gradient accumulation for effective batch size
- MLflow experiment tracking
- LOSO cross-validation
- Ablation: transformer vs. LSTM-only
"""

import os
import time
import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

from config import SAMPLE_RATE, WINDOW_SIZE, STRIDE, OUTPUT_DIR, DATASET_DIR
from data.dataset import CHBMITDataset, create_dataloader
from data.split import loso_split
from features.salt_ltformer import SALT_LTformer, LSTMOnly, create_model, count_parameters


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    gradient_accumulation_steps: int = 4,
    device: str = "cpu",
) -> dict:
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_idx, (x, y, _, _) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        use_amp = scaler is not None and device == "cuda"

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = F.cross_entropy(logits, y) / gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:
            logits = model(x)
            loss = F.cross_entropy(logits, y) / gradient_accumulation_steps
            loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())

    if scheduler is not None:
        scheduler.step()

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def eval_model(
    model: nn.Module, loader: DataLoader, device: str = "cpu"
) -> dict:
    model.eval()
    all_logits = []
    all_labels = []
    all_patients = []

    with torch.no_grad():
        for x, y, pid, _ in loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.numpy())
            all_patients.extend(pid)

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    preds = logits.argmax(axis=1)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
    }

    if len(np.unique(labels)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(
                labels, F.softmax(torch.from_numpy(logits), dim=1).numpy(),
                multi_class="ovr", average="weighted",
            )
        except Exception:
            metrics["roc_auc"] = 0.0

    return metrics, logits, labels, all_patients


def train_loso(
    n_channels: int = 18,
    num_classes: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    use_transformer: bool = True,
    use_mlflow: bool = False,
    device: str = "cpu",
):
    if use_mlflow:
        import mlflow
        mlflow.set_experiment("pomas_salt_ltformer")

    device = torch.device(device)
    print(f"Device: {device}")
    print(f"Model: {'SALT_LTformer' if use_transformer else 'LSTMOnly'}")

    ds = CHBMITDataset(preload=True)
    splits = loso_split(ds)
    results = []

    for test_patient, split_data in splits.items():
        print(f"\n{'='*40}")
        print(f"Testing: {test_patient}")
        print(f"Train: {len(split_data['train'])} | Test: {len(split_data['test'])}")

        train_loader = DataLoader(
            Subset(ds, split_data["train"]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            Subset(ds, split_data["test"]),
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=0,
        )

        model = create_model(
            use_transformer=use_transformer,
            n_channels=n_channels,
            num_classes=num_classes,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

        for epoch in range(epochs):
            train_metrics = train_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                gradient_accumulation_steps, device,
            )
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")

        test_metrics, logits, labels, patients = eval_model(model, test_loader, device)
        test_metrics["test_patient"] = test_patient
        results.append(test_metrics)
        print(f"  Test - Acc: {test_metrics['accuracy']:.4f}, "
              f"F1: {test_metrics['f1_weighted']:.4f}, "
              f"AUC: {test_metrics.get('roc_auc', 0):.4f}")

        if use_mlflow:
            with mlflow.start_run(run_name=f"loso_{test_patient}"):
                mlflow.log_params({
                    "model": "SALT_LTformer" if use_transformer else "LSTMOnly",
                    "test_patient": test_patient,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "gradient_accumulation": gradient_accumulation_steps,
                })
                mlflow.log_metrics(test_metrics)

    avg = {k: np.mean([r[k] for r in results]) for k in results[0] if k != "test_patient"}
    print(f"\n{'='*40}")
    print("Average LOSO Results:")
    for k, v in avg.items():
        print(f"  {k}: {v:.4f}")

    return results, avg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lstm-only", action="store_true", help="Use LSTM-only fallback")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_loso(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_transformer=not args.lstm_only,
        use_mlflow=args.mlflow,
        device=args.device,
    )
