"""plots.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations
import matplotlib.pyplot as plt

def plot_run(result: dict, path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    hist = result["history"]
    cfg = result["cfg"]
    epochs = hist["epoch"]
    
    # Title and configs
    title = f"{cfg.get('exp_id', 'unknown')} | Opt: {cfg.get('optimizer')}, LR: {cfg.get('lr')}, Batch: {cfg.get('batch')}"
    fig.suptitle(title)
    
    best_epoch = result.get("summary", {}).get("best_epoch", -1)
    
    # (1) Loss
    axes[0].plot(epochs, hist["train_loss"], label="Train Loss")
    axes[0].plot(epochs, hist["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    if best_epoch > 0: axes[0].axvline(best_epoch, color='r', linestyle='--', alpha=0.5)
    
    # (2) Accuracy and Macro-F1
    axes[1].plot(epochs, hist["val_acc"], label="Val Acc")
    axes[1].plot(epochs, hist.get("val_macro_f1", []), label="Val Macro F1")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    if best_epoch > 0: axes[1].axvline(best_epoch, color='r', linestyle='--', alpha=0.5)
    
    # (3) Grad Norm
    axes[2].plot(epochs, hist["grad_norm"], label="Grad Norm")
    axes[2].set_title("Gradient Norm")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for res in results:
        hist = res["history"]
        exp_id = res["cfg"].get("exp_id", "unknown")
        if metric in hist:
            ax.plot(hist["epoch"], hist[metric], label=exp_id)
    
    ax.set_title(title if title else f"Compare {metric}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.legend()
    
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
