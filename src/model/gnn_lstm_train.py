#!/usr/bin/env python3
"""
GC-LSTM fMRI classifier – CPU-only, NumPy-1.26-compatible.

* Early stopping (patience = 5) – best epoch weights kept.
* Confusion-matrix image + classification report.
* ROI-importance:
    - heat-map (class × ROI)  → roi_importance_heat.png
    - SHAP-style horizontal bar plot per class → roi_imp_class_<k>.png
      (uses lightweight grad×input saliency, NO shap dependency).

All plotting is done with plain Matplotlib – no seaborn, so it fits the
given requirements.txt (NumPy 1.26, matplotlib 3.8).
"""

import os, json, math
from pathlib import Path
from collections import defaultdict

# -----------------------------------------------------------------------------#
# Configuration                                                                #
# -----------------------------------------------------------------------------#
os.environ["CUDA_VISIBLE_DEVICES"] = ""            #〈 force CPU 〉
GRAPH_DIR = "/export/kroy/Capstone_Project-fMRI_analysis/resources/clustered_graph_dir_v1"
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE    = 5        # early-stopping patience

# -----------------------------------------------------------------------------#
# Imports (requirements-compatible)                                            #
# -----------------------------------------------------------------------------#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------#
# Helpers                                                                      #
# -----------------------------------------------------------------------------#
def set_threads():
    """Use all available physical cores without oversubscription."""
    torch.set_num_threads(os.cpu_count() or 1)

def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))

# -----------------------------------------------------------------------------#
# Dataset wrapper (drop x,y,z coords)                                          #
# -----------------------------------------------------------------------------#
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs):
        self.graphs = [g.clone() for g in graphs]
        for g in self.graphs:
            g.x = g.x[:, 3:]          # keep only fMRI features
    def __len__(self):  return len(self.graphs)
    def __getitem__(self, idx): return self.graphs[idx]

# -----------------------------------------------------------------------------#
# Model                                                                        #
# -----------------------------------------------------------------------------#
class SpatioTemporalGNN(nn.Module):
    def __init__(self, hid_lstm=128, hid_gnn=128, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(1, hid_lstm, batch_first=True,
                            bidirectional=True)
        self.gcn  = GCNConv(hid_lstm * 2, hid_gnn)
        self.dp   = nn.Dropout(0.5)
        self.cls  = nn.Linear(hid_gnn, num_classes)

    def forward(self, data):
        x, ei, ew, batch = (data.x, data.edge_index,
                            data.edge_weight, data.batch)
        T = x.size(1)                # time dimension
        x = x.unsqueeze(-1)          # [N, T, 1]

        lstm_out, _ = self.lstm(x)   # [N, T, 2H]
        feats = []
        for t in range(T):
            feats_t = lstm_out[:, t, :]
            feats.append(
                F.relu(self.gcn(feats_t, ei, ew))
            )
        node_repr = torch.stack(feats, 1).mean(1)   # [N, H]
        graph_repr = global_mean_pool(node_repr, batch)
        return self.cls(self.dp(graph_repr))

# -----------------------------------------------------------------------------#
# Graph loading & splits                                                       #
# -----------------------------------------------------------------------------#
def load_graphs(directory=GRAPH_DIR):
    label_map = {"CN": 0, "EMCI": 1, "MCI": 2}
    graphs, labels, subjects = [], [], []
    for fp in Path(directory).glob("*_bold_clustered.pt"):
        parts = fp.name.split("_")
        subjects.append("_".join(parts[:3]))
        labels.append(label_map[parts[4]])
        graphs.append(torch.load(fp))
    return graphs, labels, subjects

def split_sets(graphs, labels, subjects):
    subj_to_idx = {s: i for i, s in enumerate(sorted(set(subjects)))}
    groups = np.array([subj_to_idx[s] for s in subjects])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_val_idx, test_idx = next(gss.split(graphs, groups=groups))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, val_idx = next(
        gss2.split([graphs[i] for i in tr_val_idx],
                   groups=groups[tr_val_idx])
    )

    train = [graphs[tr_val_idx[i]] for i in tr_idx]
    val   = [graphs[tr_val_idx[i]] for i in val_idx]
    test  = [graphs[i] for i in test_idx]
    return train, val, test

def group_by_T(graphs):
    buckets = defaultdict(list)
    for g in graphs:
        buckets[g.x.size(1) - 3].append(g)
    return buckets

def make_loaders(buckets, shuffle=False):
    return {
        T: DataLoader(
            GraphDataset(lst), batch_size=BATCH_SIZE,
            shuffle=shuffle, num_workers=0, pin_memory=False
        )
        for T, lst in buckets.items()
    }

# -----------------------------------------------------------------------------#
# Evaluation helpers                                                           #
# -----------------------------------------------------------------------------#
@torch.inference_mode()
def evaluate(model, loaders, device):
    model.eval()
    preds, targets = [], []
    for dl in loaders.values():
        for batch in dl:
            batch = batch.to(device)
            preds .append(model(batch).argmax(1).cpu())
            targets.append(batch.y.cpu())
    if not targets:
        return 0.0, np.array([]), np.array([])
    preds = torch.cat(preds).numpy()
    true  = torch.cat(targets).numpy()
    return (preds == true).mean(), preds, true

@torch.no_grad()
def val_loss(model, loaders, device, criterion):
    model.eval()
    total = n = 0
    for dl in loaders.values():
        for batch in dl:
            batch = batch.to(device)
            total += criterion(model(batch), batch.y).item()
            n += 1
    return total / n if n else math.inf

# -----------------------------------------------------------------------------#
# ROI-importance via grad×input                                                #
# -----------------------------------------------------------------------------#
def roi_importance(model, graphs, device):
    model.eval()
    by_lbl = defaultdict(list)
    for g in graphs:
        by_lbl[g.y.item()].append(g)

    imp = defaultdict(lambda: defaultdict(list))
    for lbl, g_list in by_lbl.items():
        for g in g_list:
            g = g.clone().to(device)
            g.x.requires_grad_(True)
            prob = F.softmax(model(g), dim=1)[0, lbl]
            prob.backward()
            sal = (g.x.grad.abs() * g.x.detach()).mean(1).cpu().numpy()
            roi_ids = g.roi.cpu().numpy()       # assumes .roi is present
            for r, s in zip(roi_ids, sal):
                imp[lbl][int(r)].append(float(s))

    return {
        c: {r: float(np.mean(v)) for r, v in d.items()}
        for c, d in imp.items()
    }

# -----------------------------------------------------------------------------#
# Matplotlib visualisations (no seaborn)                                       #
# -----------------------------------------------------------------------------#
def plot_confusion(cm, labels, outfile):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]),
                    ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(outfile, dpi=300); plt.close(fig)
    print(f"[INFO] confusion matrix → {outfile}")

def plot_heatmap(roi_imp, outfile):
    classes = sorted(roi_imp)
    rois    = sorted({r for d in roi_imp.values() for r in d})
    mat = np.zeros((len(classes), len(rois)))
    for i, c in enumerate(classes):
        for j, r in enumerate(rois):
            mat[i, j] = roi_imp[c].get(r, 0.0)

    fig, ax = plt.subplots(figsize=(0.5*len(rois)+1, 2+0.4*len(classes)))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(range(len(rois))); ax.set_xticklabels([str(r) for r in rois],
                                                        rotation=90)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(
        [f"Class {c}" for c in classes])
    ax.set_xlabel("ROI ID"); ax.set_ylabel("Class")
    ax.set_title("ROI importance (grad×input)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(outfile, dpi=300); plt.close(fig)
    print(f"[INFO] ROI heat-map → {outfile}")

def plot_bar(roi_imp_class, cls_idx):
    if not roi_imp_class: return
    rois, scores = zip(*sorted(roi_imp_class.items(),
                               key=lambda x: -x[1]))
    y = np.arange(len(rois))
    fig, ax = plt.subplots(figsize=(6, 0.3*len(rois)+1))
    ax.barh(y, scores, align="center")
    ax.set_yticks(y); ax.set_yticklabels([str(r) for r in rois])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Class {cls_idx} – ROI importance")
    fig.tight_layout()
    fn = f"roi_imp_class_{cls_idx}.png"
    fig.savefig(fn, dpi=300); plt.close(fig)
    print(f"[INFO] SHAP-style bar plot → {fn}")

# -----------------------------------------------------------------------------#
# Main                                                                         #
# -----------------------------------------------------------------------------#
def main():
    set_threads()
    device = torch.device("cpu")
    print(f"[INFO] CPU threads: {torch.get_num_threads()}")

    graphs, labels, subjects = load_graphs()
    print(f"[INFO] Loaded {len(graphs)} graphs")

    train_g, val_g, test_g = split_sets(graphs, labels, subjects)
    print(f"[INFO] Split  Train={len(train_g)}  Val={len(val_g)}  "
          f"Test={len(test_g)}")

    train_L = make_loaders(group_by_T(train_g), shuffle=True)
    val_L   = make_loaders(group_by_T(val_g))
    test_L  = make_loaders(group_by_T(test_g))

    model = SpatioTemporalGNN().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                        patience=2, factor=0.5,
                                                        verbose=True)
    crit   = nn.CrossEntropyLoss()

    best_val  = math.inf
    no_improv = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tot, n = 0.0, 0
        for dl in train_L.values():
            for batch in dl:
                batch = batch.to(device)
                opt.zero_grad()
                loss = crit(model(batch), batch.y)
                loss.backward()
                opt.step()
                tot += loss.item(); n += 1
        train_loss = tot / n

        v_loss = val_loss(model, val_L, device, crit)
        sched.step(v_loss)
        print(f"Ep {epoch:03d} | train {train_loss:.4f} | val {v_loss:.4f}")

        if v_loss < best_val - 1e-4:
            best_val = v_loss
            no_improv = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            no_improv += 1
            if no_improv >= PATIENCE:
                print(f"[INFO] early-stopping at epoch {epoch}")
                break

    # ----------------------------------------------------------------- eval
    model.load_state_dict(torch.load("best_model.pt"))
    acc, preds, true = evaluate(model, test_L, device)
    print(f"[RESULT] Test accuracy = {acc:.4f}")
    print(classification_report(true, preds,
                                target_names=["CN", "EMCI", "MCI"]))

    cm = confusion_matrix(true, preds, labels=[0,1,2])
    plot_confusion(cm, ["CN", "EMCI", "MCI"], "confusion_matrix.png")

    # ------------------------------------------------ ROI importance & plots
    roi_imp = roi_importance(model, test_g, device)
    save_json(roi_imp, "roi_importance.json")
    plot_heatmap(roi_imp, "roi_importance_heat.png")
    for cls_idx, imp in roi_imp.items():
        plot_bar(imp, cls_idx)

if __name__ == "__main__":
    main()
