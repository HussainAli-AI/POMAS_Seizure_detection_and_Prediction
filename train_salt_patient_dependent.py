"""Patient-dependent three-class SALT LTformer experiment for one CHB-MIT subject."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import OUTPUT_DIR
from data.hdf5_dataset import HDF5WindowDataset
from features.salt_ltformer import SALT_LTformer
from train_salt_detector import evaluate


def split_indices(path, validation_recordings, test_recordings, interictal_ratio, seed=42):
    with h5py.File(path, "r") as f:
        labels=f["labels"][:]; ids=np.asarray(f["recording_ids"][:], dtype=str)
    validation=np.flatnonzero(np.isin(ids, validation_recordings))
    test=np.flatnonzero(np.isin(ids, test_recordings))
    train=np.flatnonzero(~np.isin(ids, validation_recordings + test_recordings))
    positives=train[labels[train] != 0]; interictal=train[labels[train] == 0]
    rng=np.random.default_rng(seed); limit=min(len(interictal), int(interictal_ratio * len(positives)))
    train=np.sort(np.concatenate([positives, rng.choice(interictal, size=limit, replace=False)]))
    for name, indices in (("train",train),("validation",validation),("test",test)):
        if len(np.unique(labels[indices])) < 3: raise ValueError(f"{name} lacks a required class: {np.unique(labels[indices])}")
    return train, validation, test, labels


def run(data, validation_recordings, test_recordings, epochs, batch_size, interictal_ratio):
    train_i,val_i,test_i,labels=split_indices(data,validation_recordings,test_recordings,interictal_ratio)
    train=HDF5WindowDataset(data,train_i); val=HDF5WindowDataset(data,val_i); test=HDF5WindowDataset(data,test_i)
    counts=np.bincount(labels[train_i],minlength=3); weights=torch.tensor(counts.sum()/(3*np.maximum(counts,1)),dtype=torch.float32)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=SALT_LTformer().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4); criterion=nn.CrossEntropyLoss(weight=weights.to(device)); scaler=torch.amp.GradScaler(device.type,enabled=device.type=="cuda")
    train_loader=DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=device.type=="cuda"); val_loader=DataLoader(val,batch_size=batch_size*2,shuffle=False,num_workers=0); test_loader=DataLoader(test,batch_size=batch_size*2,shuffle=False,num_workers=0)
    print(f"device={device}; train={len(train)} validation={len(val)} test={len(test)} class_counts={counts.tolist()}")
    best=(-1,None)
    for epoch in range(1,epochs+1):
        model.train(); total=0
        for x,y,_ in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,enabled=device.type=="cuda"): loss=criterion(model(x.to(device)),y.to(device))
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();total+=loss.item()
        m=evaluate(model,val_loader,device);print(f"epoch={epoch}/{epochs} loss={total/len(train_loader):.4f} val_macro_f1={m['macro_f1']:.4f} val_preictal_recall={m['preictal_recall']:.4f}")
        if m["macro_f1"]>best[0]:best=(m["macro_f1"],{k:v.detach().cpu() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); results=evaluate(model,test_loader,device);results.update({"validation_recordings":validation_recordings,"test_recordings":test_recordings,"best_validation_macro_f1":best[0],"interictal_ratio":interictal_ratio})
    torch.save({"model_state":model.state_dict(),"results":results},OUTPUT_DIR/"salt_ltformer_chb01_patient_dependent.pt");(OUTPUT_DIR/"salt_ltformer_chb01_patient_dependent_results.json").write_text(json.dumps(results,indent=2),encoding="utf-8");print(json.dumps(results,indent=2))
    for d in (train,val,test):d.close()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,default=OUTPUT_DIR/"preprocessed"/"chb01_preprocessed.h5");p.add_argument("--validation-recordings",nargs="+",default=["chb01_18.edf"]);p.add_argument("--test-recordings",nargs="+",default=["chb01_21.edf","chb01_26.edf"]);p.add_argument("--epochs",type=int,default=10);p.add_argument("--batch-size",type=int,default=4);p.add_argument("--interictal-ratio",type=float,default=3.0);a=p.parse_args();run(a.data,a.validation_recordings,a.test_recordings,a.epochs,a.batch_size,a.interictal_ratio)
