"""Three-class, patient-independent SALT LTformer detector experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from config import OUTPUT_DIR
from data.hdf5_dataset import HDF5WindowDataset
from features.salt_ltformer import SALT_LTformer


def datasets(data_dir, subjects):
    return [HDF5WindowDataset(Path(data_dir) / f"{s}_preprocessed.h5") for s in subjects]


def evaluate(model, loader, device):
    model.eval(); truth=[]; predicted=[]
    with torch.no_grad():
        for x, y, _ in loader:
            predicted.extend(model(x.to(device)).argmax(1).cpu().numpy())
            truth.extend(y.numpy())
    return {"accuracy": float(accuracy_score(truth, predicted)), "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)), "preictal_recall": float(recall_score(truth, predicted, labels=[1], average=None, zero_division=0)[0])}


def run(data_dir, train_subjects, validation_subject, test_subject, epochs, batch_size):
    train_parts=datasets(data_dir, train_subjects); val_parts=datasets(data_dir, [validation_subject]); test_parts=datasets(data_dir, [test_subject])
    train=ConcatDataset(train_parts); val=ConcatDataset(val_parts); test=ConcatDataset(test_parts)
    labels=np.concatenate([d.labels for d in train_parts]); counts=np.bincount(labels, minlength=3)
    weights=torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=SALT_LTformer().to(device); opt=torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion=nn.CrossEntropyLoss(weight=weights.to(device)); scaler=torch.amp.GradScaler(device.type, enabled=device.type=="cuda")
    train_loader=DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type=="cuda")
    val_loader=DataLoader(val, batch_size=batch_size*2, shuffle=False, num_workers=0); test_loader=DataLoader(test, batch_size=batch_size*2, shuffle=False, num_workers=0)
    print(f"device={device}; train={len(train)} validation={len(val)} test={len(test)} class_counts={counts.tolist()}")
    best=(-1.0, None)
    for epoch in range(1, epochs+1):
        model.train(); total=0.0
        for x,y,_ in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type=="cuda"):
                loss=criterion(model(x.to(device, non_blocking=True)), y.to(device, non_blocking=True))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total+=loss.item()
        metrics=evaluate(model, val_loader, device)
        print(f"epoch={epoch}/{epochs} loss={total/len(train_loader):.4f} val_macro_f1={metrics['macro_f1']:.4f} val_preictal_recall={metrics['preictal_recall']:.4f}")
        if metrics["macro_f1"] > best[0]: best=(metrics["macro_f1"], {k:v.detach().cpu() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); results=evaluate(model, test_loader, device); results.update({"train_subjects":train_subjects,"validation_subject":validation_subject,"test_subject":test_subject,"best_validation_macro_f1":best[0]})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True); torch.save({"model_state":model.state_dict(),"results":results}, OUTPUT_DIR/"salt_ltformer_detector.pt")
    (OUTPUT_DIR/"salt_ltformer_detector_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8"); print(json.dumps(results,indent=2))
    for d in train_parts+val_parts+test_parts: d.close()


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,default=OUTPUT_DIR/"preprocessed"); p.add_argument("--train-subjects",nargs="+",default=["chb01","chb02"]); p.add_argument("--validation-subject",default="chb03"); p.add_argument("--test-subject",default="chb05"); p.add_argument("--epochs",type=int,default=10); p.add_argument("--batch-size",type=int,default=4); a=p.parse_args(); run(a.data_dir,a.train_subjects,a.validation_subject,a.test_subject,a.epochs,a.batch_size)
