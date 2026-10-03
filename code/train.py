"""train.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import time
import random
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from data import iterate_batches
from model import MLP, count_params
from optimizer import build_optimizer, clip_gradients

DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",
    optimizer="sgd_momentum",
    lr=0.01,
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,
    precision="fp32",
    seed=1,
)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def macro_f1_from_confusion(cm: np.ndarray) -> float:
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    
    with np.errstate(divide='ignore', invalid='ignore'):
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        
    f1[np.isnan(f1)] = 0.0
    return float(np.mean(f1))

@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    model.eval()
    preds = []
    N = len(X)
    for i in range(0, N, batch_size):
        xb = X[i:i+batch_size]
        logits = model(xb)
        preds.append(torch.argmax(logits, dim=1))
    return torch.cat(preds)

@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    model.eval()
    total_loss = 0.0
    preds = []
    N = len(X)
    
    for i in range(0, N, batch_size):
        xb = X[i:i+batch_size]
        yb = y[i:i+batch_size]
        logits = model(xb)
        
        loss = compute_loss(logits, yb, loss_name)
        total_loss += loss.item() * len(yb)
        preds.append(torch.argmax(logits, dim=1))
        
    mean_loss = total_loss / N
    all_preds = torch.cat(preds)
    acc = (all_preds == y).float().mean().item()
    
    cm = confusion_matrix(y.cpu().numpy(), all_preds.cpu().numpy(), labels=range(7))
    macro_f1 = macro_f1_from_confusion(cm)
    
    return {"loss": mean_loss, "acc": acc, "macro_f1": macro_f1}

def compute_loss(logits, y, loss_name: str):
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_onehot = F.one_hot(y, num_classes=logits.size(1)).float()
        return F.mse_loss(logits, y_onehot)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

def run_experiment(cfg: dict, data: dict) -> dict:
    set_seed(cfg["seed"])
    
    X_tr, y_tr = data["X_tr"], data["y_tr"]
    X_val, y_val = data["X_val"], data["y_val"]
    
    model = MLP(hidden=cfg["hidden"], dropout=cfg["dropout"], init=cfg["init"], in_features=X_tr.size(1), num_classes=7)
    model = model.to(X_tr.device)
    
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"], 
                                weight_decay=cfg.get("weight_decay", 0.0), momentum=cfg.get("momentum", 0.9))
    
    precision = cfg.get("precision", "fp32")
    scaler = torch.amp.GradScaler(enabled=(precision == "fp16"))
    
    step0_res = evaluate(model, X_val, y_val, cfg["loss"])
    step0_loss = step0_res["loss"]
    
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "grad_norm": [], "epoch_time_s": []}
    
    best_val_loss = float('inf')
    best_epoch = -1
    best_state = None
    diverged = False
    
    epochs = cfg.get("epochs", 20)
    batch_size = cfg.get("batch", 512)
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_grad_norms = []
        start_t = time.time()
        
        for xb, yb in iterate_batches(X_tr, y_tr, batch_size):
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type="cuda" if "cuda" in str(X_tr.device) else "cpu", enabled=(precision != "fp32"), dtype=torch.float16 if precision == "fp16" else torch.bfloat16):
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])
                
            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break
                
            scaler.scale(loss).backward()
            
            if precision == "fp16" and cfg.get("clip_norm"):
                scaler.unscale_(optimizer)
                
            gn = clip_gradients(model.parameters(), cfg.get("clip_norm"))
            epoch_grad_norms.append(gn)
            
            scaler.step(optimizer)
            scaler.update()
            
        if diverged:
            print(f"Epoch {epoch}: Diverged!")
            break
            
        if torch.cuda.is_available(): torch.cuda.synchronize()
        end_t = time.time()
        
        val_res = evaluate(model, X_val, y_val, cfg["loss"])
        # approximate train loss using validation batch evaluate to save time
        train_res = evaluate(model, X_tr[:50000], y_tr[:50000], cfg["loss"]) 
        
        epoch_gn = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0
        
        hist["epoch"].append(epoch)
        hist["train_loss"].append(train_res["loss"])
        hist["val_loss"].append(val_res["loss"])
        hist["val_acc"].append(val_res["acc"])
        hist["val_macro_f1"].append(val_res["macro_f1"])
        hist["grad_norm"].append(epoch_gn)
        hist["epoch_time_s"].append(end_t - start_t)
        
        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": hist["train_loss"][-1] if hist["train_loss"] else None,
        "final_val_loss": hist["val_loss"][-1] if hist["val_loss"] else None,
        "val_acc": hist["val_acc"][best_epoch-1] if best_epoch > 0 else None,
        "val_macro_f1": hist["val_macro_f1"][best_epoch-1] if best_epoch > 0 else None,
        "time_per_epoch_s": np.mean(hist["epoch_time_s"]) if hist["epoch_time_s"] else None,
        "peak_mem_MB": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0,
        "diverged": diverged
    }
    
    return {"cfg": cfg, "history": hist, "summary": summary, "best_state": best_state}

def write_predictions(row_id, preds, path: str) -> None:
    with open(path, "w") as f:
        f.write("row_id,pred\n")
        for rid, p in zip(row_id, preds):
            f.write(f"{rid},{p}\n")

def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    model = MLP(hidden=cfg["hidden"], dropout=cfg["dropout"], init=cfg["init"], in_features=data["X_eval"].size(1), num_classes=7)
    model.load_state_dict(result["best_state"])
    model = model.to(data["X_eval"].device)
    
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    
    subprocess.run(["python", "scripts/evaluate.py", "--pred", pred_path])
