"""Fine-tune official LaBraM Base weights for three-class seizure detection."""
import argparse, json
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score
from config import OUTPUT_DIR
from data.hdf5_dataset import HDF5WindowDataset
from models.labram_loader import build_labram_base, labram_input_from_pomas

def evaluate(model, loader, device):
    model.eval(); y=[]; p=[]
    with torch.no_grad():
        for x,t,_ in loader:
            x,ch=labram_input_from_pomas(x); p.extend(model(x.to(device),input_chans=ch).argmax(1).cpu().numpy()); y.extend(t.numpy())
    return {"accuracy":float(accuracy_score(y,p)),"macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),"preictal_recall":float(recall_score(y,p,labels=[1],average=None,zero_division=0)[0])}

def run(data_dir, train_subjects, validation_subject, test_subject, epochs, batch_size, lr):
    train_parts=[HDF5WindowDataset(data_dir/f"{s}_labram_200hz.h5") for s in train_subjects]
    val_parts=[HDF5WindowDataset(data_dir/f"{validation_subject}_labram_200hz.h5")]; test_parts=[HDF5WindowDataset(data_dir/f"{test_subject}_labram_200hz.h5")]
    train=ConcatDataset(train_parts); val=ConcatDataset(val_parts); test=ConcatDataset(test_parts)
    labels=np.concatenate([d.labels for d in train_parts]); counts=np.bincount(labels,minlength=3); weights=torch.tensor(counts.sum()/(3*np.maximum(counts,1)),dtype=torch.float32)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,load_info=build_labram_base(Path("models/pretrained/labram-base.pth")); model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.05); loss_fn=nn.CrossEntropyLoss(weight=weights.to(device)); scaler=torch.amp.GradScaler(device.type,enabled=device.type=="cuda")
    train_loader=DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=device.type=="cuda"); val_loader=DataLoader(val,batch_size=batch_size*2,shuffle=False,num_workers=0); test_loader=DataLoader(test,batch_size=batch_size*2,shuffle=False,num_workers=0)
    print(f"device={device}; pretrained_tensors={load_info['loaded_tensors']}; train={len(train)} validation={len(val)} test={len(test)}")
    best=(-1,None)
    for epoch in range(1,epochs+1):
        model.train(); total=0
        for x,y,_ in train_loader:
            x,ch=labram_input_from_pomas(x); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,enabled=device.type=="cuda"): loss=loss_fn(model(x.to(device),input_chans=ch),y.to(device))
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();total+=loss.item()
        m=evaluate(model,val_loader,device); print(f"epoch={epoch}/{epochs} loss={total/len(train_loader):.4f} val_macro_f1={m['macro_f1']:.4f} val_preictal_recall={m['preictal_recall']:.4f}")
        if m['macro_f1']>best[0]: best=(m['macro_f1'],{k:v.detach().cpu() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); results=evaluate(model,test_loader,device);results.update({"train_subjects":train_subjects,"validation_subject":validation_subject,"test_subject":test_subject,"best_validation_macro_f1":best[0],"pretrained_tensors_loaded":load_info['loaded_tensors']})
    torch.save({"model_state":model.state_dict(),"results":results},OUTPUT_DIR/"labram_detector.pt");(OUTPUT_DIR/"labram_detector_results.json").write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
    for d in train_parts+val_parts+test_parts:d.close()

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data-dir",type=Path,default=OUTPUT_DIR/"labram_preprocessed");p.add_argument("--train-subjects",nargs="+",default=["chb01","chb02"]);p.add_argument("--validation-subject",default="chb03");p.add_argument("--test-subject",default="chb05");p.add_argument("--epochs",type=int,default=10);p.add_argument("--batch-size",type=int,default=8);p.add_argument("--lr",type=float,default=1e-5);a=p.parse_args();run(a.data_dir,a.train_subjects,a.validation_subject,a.test_subject,a.epochs,a.batch_size,a.lr)
