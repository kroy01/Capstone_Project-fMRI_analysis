#!/usr/bin/env python3
"""
Predict a single fMRI graph with a trained GC-LSTM model and visualise
ROI importance (grad×input saliency).

Outputs
-------
<graph>_prediction.json   – {"pred": class_id, "probs": [p_CN, p_EMCI, p_MCI]}
<graph>_roi_imp.png       – horizontal bar chart of ROI importance
"""

import os, json, argparse
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""      # CPU only

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt

# ----------------------------- model definition ------------------------------ #
class SpatioTemporalGNN(nn.Module):
    def __init__(self, hid_lstm=128, hid_gnn=128, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(1, hid_lstm, batch_first=True, bidirectional=True)
        self.gcn  = GCNConv(hid_lstm*2, hid_gnn)
        self.dp   = nn.Dropout(0.5)
        self.cls  = nn.Linear(hid_gnn, num_classes)
    def forward(self, data):
        x, ei, ew, batch = (data.x, data.edge_index,
                            data.edge_weight, data.batch)
        T = x.size(1)
        x = x.unsqueeze(-1)                       # [N,T,1]
        lstm_out, _ = self.lstm(x)                # [N,T,2H]
        feats = []
        for t in range(T):
            feats_t = lstm_out[:, t, :]
            feats.append(F.relu(self.gcn(feats_t, ei, ew)))
        node_repr = torch.stack(feats, 1).mean(1) # [N,H]
        g_repr = global_mean_pool(node_repr, batch)
        return self.cls(self.dp(g_repr))          # [1,C]

# ---------------------------- ROI saliency plot ------------------------------ #
def plot_roi_imp(roi_imp, outfile):
    if not roi_imp:
        print("[WARN] No ROI importance computed (empty roi_imp dict).")
        return
    rois, scores = zip(*sorted(roi_imp.items(), key=lambda x: -x[1]))
    y = np.arange(len(rois))
    fig, ax = plt.subplots(figsize=(6, 0.3*len(rois)+1))
    ax.barh(y, scores, align="center")
    ax.set_yticks(y); ax.set_yticklabels([str(r) for r in rois])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("ROI importance (grad×input)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"[INFO] ROI bar plot saved → {outfile}")

# ----------------------------- main routine ---------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph", help="Path to *_bold_clustered.pt graph file")
    ap.add_argument("--weights", default="best_model.pt",
                    help="Trained model weights (default: best_model.pt)")
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 1)

    # -------- load graph -------- #
    graph_path = Path(args.graph)
    if not graph_path.exists():
        raise FileNotFoundError(graph_path)
    g = torch.load(graph_path)
    g.x = g.x[:, 3:]              # drop xyz coords
    g = g.to(device)

    # -------- load model -------- #
    model = SpatioTemporalGNN().to(device)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    # -------- prediction -------- #
    with torch.inference_mode():
        logits = model(g)
        probs  = F.softmax(logits, 1).cpu().numpy()[0]   # [3]
        pred   = int(np.argmax(probs))

    # save prediction JSON
    pred_out = graph_path.with_suffix("").as_posix() + "_prediction.json"
    with open(pred_out, "w") as f:
        json.dump({"pred": pred, "probs": probs.tolist()}, f, indent=2)
    print(f"[INFO] Prediction saved → {pred_out}")

    # -------- ROI importance (grad×input) -------- #
    g.x.requires_grad_(True)
    logits = model(g)                       # forward pass
    prob_pred = F.softmax(logits, 1)[0, pred]
    prob_pred.backward()
    sal = (g.x.grad.abs() * g.x.detach()).mean(1).cpu().numpy()  # [N]
    roi_ids = g.roi.cpu().numpy()                                # [N]

    roi_imp = {}
    for r, s in zip(roi_ids, sal):
        roi_imp[int(r)] = roi_imp.get(int(r), 0.0) + float(s)
    # average if multiple nodes per ROI
    counts = np.bincount(roi_ids)
    for r in roi_imp:
        if counts[r] > 0:
            roi_imp[r] /= counts[r]

    # plot
    bar_out = graph_path.with_suffix("").as_posix() + "_roi_imp.png"
    plot_roi_imp(roi_imp, bar_out)

if __name__ == "__main__":
    main()
