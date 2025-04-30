#!/usr/bin/env python3
import argparse
import os
import nibabel as nib
import numpy as np
import torch
from torch_geometric.data import Data
from nibabel.processing import resample_from_to
from scipy.stats import pearsonr

def load_fmri_graph_bold_with_coords(
        fmri_path: str,
        label_path: str,
        graph_label: int = 0,
        threshold: float = 1e-6,
        corr_threshold: float = 0.3,
        max_distance: float = 10.0,
        roi_ids: list[int] | None = None
) -> Data:
    # Load images
    fmri_img = nib.load(fmri_path)
    data_4d = fmri_img.get_fdata()  # (X, Y, Z, T)
    label_img = nib.load(label_path)

    # Resample label volume if needed
    if label_img.shape != fmri_img.shape[:3] or not np.allclose(label_img.affine, fmri_img.affine):
        print(f"Resampling label map from {label_img.shape} → {fmri_img.shape[:3]}")
        label_img = resample_from_to(
            label_img,
            (fmri_img.shape[:3], fmri_img.affine),
            order=0  # nearest‐neighbor for labels
        )
    labels_3d = label_img.get_fdata().astype(int)  # (X, Y, Z)

    # Brain mask on mean‐BOLD
    mean_3d = data_4d.mean(axis=3)
    mask = mean_3d > threshold
    coords = np.argwhere(mask)  # [[x, y, z], …]

    # Node features and ROI labels
    feats = []
    rois = []
    for (x, y, z) in coords:
        ts = data_4d[x, y, z, :]
        feats.append(np.concatenate(([x, y, z], ts)))
        rois.append(labels_3d[x, y, z])
    feats = np.array(feats, dtype=np.float32)
    rois = np.array(rois, dtype=np.int64)

    # Early ROI filtering
    if roi_ids is not None:
        keep = np.isin(rois, roi_ids)
        feats = feats[keep]
        rois  = rois[keep]
        coords = coords[keep]
        if feats.size == 0:
            raise ValueError(f"No nodes left after ROI filtering → {roi_ids!r}")

    unique_rois = np.unique(rois)
    print(f"  → found {len(unique_rois)} unique ROIs in mask: {unique_rois}")

    # Build edges based on functional connectivity
    ts_mat = feats[:, 3:]  # (N, T)
    edges = []
    weights = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= max_distance:
                r, _ = pearsonr(ts_mat[i], ts_mat[j])
                if abs(r) >= corr_threshold:
                    w = (r + 1.0) / 2.0
                    edges += [[i, j], [j, i]]
                    weights += [w, w]

    if edges:
        edges = np.array(edges, dtype=np.int64).T
        weights = np.array(weights, dtype=np.float32)
        wmin, wmax = weights.min(), weights.max()
        if wmax > wmin:
            weights = (weights - wmin) / (wmax - wmin)
    else:
        edges = np.zeros((2, 0), dtype=np.int64)
        weights = np.zeros((0,), dtype=np.float32)

    # Create Data object
    return Data(
        x=torch.tensor(feats, dtype=torch.float),
        edge_index=torch.tensor(edges, dtype=torch.long),
        edge_weight=torch.tensor(weights, dtype=torch.float),
        y=torch.tensor([graph_label], dtype=torch.long),
        roi=torch.tensor(rois, dtype=torch.long)
    )

def main():
    p = argparse.ArgumentParser(
        description="Build GNN graph from 4D fMRI + aparc+aseg labels"
    )
    p.add_argument("--fmri_path",      required=True, help="4D fMRI NIfTI (in MNI).")
    p.add_argument("--label_path",     required=True, help="3D label NIfTI (aparc+aseg).")
    p.add_argument("--graph_label",    type=int,   default=0,      help="Global graph label.")
    p.add_argument("--threshold",      type=float, default=1e-6,   help="Mean‑BOLD mask threshold.")
    p.add_argument("--corr_threshold", type=float, default=0.3,    help="Correlation threshold for edges.")
    p.add_argument("--max_distance",   type=float, default=10.0,   help="Max spatial distance for edges (mm).")
    p.add_argument("--roi_ids",        type=str,   default=None,
                   help="Comma‑sep ROI IDs to keep (e.g. '17,18,53').")
    p.add_argument("--out_path",       type=str,   default="bold_graph.pt",
                   help="Where to save the PyG Data object.")
    args = p.parse_args()

    # Parse ROI list if given
    roi_ids = [int(x) for x in args.roi_ids.split(",")] if args.roi_ids else None

    data = load_fmri_graph_bold_with_coords(
        fmri_path      = args.fmri_path,
        label_path     = args.label_path,
        graph_label    = args.graph_label,
        threshold      = args.threshold,
        corr_threshold = args.corr_threshold,
        max_distance   = args.max_distance,
        roi_ids        = roi_ids
    )

    print(f"Nodes: {data.num_nodes}")
    print(f"Node‑feature shape: {tuple(data.x.shape)} (coords + T)")
    print(f"Edges: {data.num_edges}")
    print(f"Graph label: {data.y.tolist()}")
    print(f"Unique ROIs: {len(torch.unique(data.roi))}")

    # Normalize the out_path (removes any duplicate slashes)
    save_path = os.path.abspath(args.out_path)
    print(f"Saving to {save_path}")
    torch.save(data, save_path)

if __name__ == "__main__":
    main()
