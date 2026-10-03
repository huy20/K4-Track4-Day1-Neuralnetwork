"""results_table.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (đừng gõ tay hàng chục dòng, rất dễ sai).

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, đừng ghi đè)
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import openpyxl

def save_result(result: dict, results_dir: str = "../results") -> str:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    exp_id = result["cfg"].get("exp_id", "unknown")
    out_path = os.path.join(results_dir, f"{exp_id}.json")
    
    out_data = {
        "cfg": result["cfg"],
        "history": result["history"],
        "summary": result["summary"]
    }
    
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=4)
        
    return out_path

def load_results(results_dir: str = "../results") -> list[dict]:
    results = []
    p = Path(results_dir)
    if not p.exists(): return results
    
    for fpath in sorted(p.glob("*.json")):
        with open(fpath, "r") as f:
            data = json.load(f)
            results.append(data)
    return results

def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    row = {}
    cfg = result["cfg"]
    summary = result.get("summary", {})
    
    for k in cfg: row[k] = cfg[k]
    for k in summary:
        if k not in row:
            row[k] = summary[k]
            
    if eval_scores:
        row["eval_acc"] = eval_scores.get("accuracy")
        row["eval_macro_f1"] = eval_scores.get("macro_f1")
    else:
        row["eval_acc"] = None
        row["eval_macro_f1"] = None
        
    row["figure_file"] = f"figures/{cfg.get('exp_id', 'unknown')}.png"
    row["notes"] = notes
    return row

def write_xlsx(rows: list[dict], template_path: str, out_path: str) -> None:
    wb = openpyxl.load_workbook(template_path)
    ws = wb["Experiments"]
    
    # Read headers from row 1
    headers = {}
    for col_idx, cell in enumerate(ws[1], start=1):
        if cell.value:
            headers[cell.value] = col_idx
            
    # Write data starting from row 2
    for r_idx, row_dict in enumerate(rows, start=2):
        for k, v in row_dict.items():
            if k in headers:
                c_idx = headers[k]
                if isinstance(v, (list, tuple)):
                    v = str(v)
                ws.cell(row=r_idx, column=c_idx, value=v)
                
    wb.save(out_path)
