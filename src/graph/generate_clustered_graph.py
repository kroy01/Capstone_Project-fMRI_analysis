#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

def cluster_graph_by_roi(
    data: Data,
    voxels_per_cluster: int = 100,
    min_clusters: int = 1,
    max_clusters: int = 50,
    corr_threshold: float = 0.3,
    max_distance: float = 10.0
):
    feats = data.x.numpy()  # (N, 3+T)
    coords = feats[:, :3]
    ts = feats[:, 3:]
    rois = data.roi.numpy()

    # Validate input
    if len(rois) == 0:
        print("Warning: No nodes in graph, returning empty Data object")
        return Data(x=torch.zeros((0, feats.shape[1]), dtype=torch.float),
                   edge_index=torch.zeros((2, 0), dtype=torch.long),
                   edge_weight=torch.zeros((0,), dtype=torch.float),
                   y=data.y.clone(),
                   roi=torch.zeros((0,), dtype=torch.long)), {}

    # Normalize BOLD time-series
    ts_min, ts_max = ts.min(), ts.max()
    ts_norm = (ts - ts_min) / (ts_max - ts_min) if ts_max > ts_min else np.zeros_like(ts)

    unique_rois = np.unique(rois)
    assignment = np.empty(len(rois), dtype=int)
    cluster_feats = []
    cluster_rois = []
    mapping = {}
    cid = 0

    # Dynamic clustering
    for roi in unique_rois:
        idx = np.where(rois == roi)[0]
        if len(idx) == 0:
            continue
        num_voxels = len(idx)
        k = max(min_clusters, min(max_clusters, num_voxels // voxels_per_cluster + 1))
        print(f"ROI {roi}: {num_voxels} voxels, clustering into {k} clusters")
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(ts_norm[idx])

        for cl in range(k):
            members = idx[labels == cl].tolist()
            mapping[cid] = members
            centroid = coords[members].mean(axis=0)
            mean_ts = ts_norm[members].mean(axis=0)
            cluster_feats.append(np.concatenate([centroid, mean_ts]).astype(np.float32))
            cluster_rois.append(int(roi))
            for m in members:
                assignment[m] = cid
            cid += 1

    # Build new node features
    x_new = torch.tensor(np.stack(cluster_feats, axis=0), dtype=torch.float) if cluster_feats else torch.zeros((0, feats.shape[1]), dtype=torch.float)
    roi_new = torch.tensor(cluster_rois, dtype=torch.long) if cluster_rois else torch.zeros((0,), dtype=torch.long)

    # Re-compute edges using KD-tree
    if len(x_new) > 0:
        centroids = x_new[:, :3].numpy()
        tree = cKDTree(centroids)
        pairs = tree.query_pairs(r=max_distance, output_type='ndarray')
        if pairs.size == 0:
            edges = np.zeros((2, 0), dtype=np.int64)
            weights = np.array([], dtype=np.float32)
        else:
            ts_new = x_new[:, 3:].numpy()
            ts_i = ts_new[pairs[:, 0]]
            ts_j = ts_new[pairs[:, 1]]
            ts_i = (ts_i - ts_i.mean(axis=1, keepdims=True)) / ts_i.std(axis=1, keepdims=True)
            ts_j = (ts_j - ts_j.mean(axis=1, keepdims=True)) / ts_j.std(axis=1, keepdims=True)
            corr = np.sum(ts_i * ts_j, axis=1) / (ts_i.shape[1] - 1)
            mask = np.abs(corr) >= corr_threshold
            edges = pairs[mask].T
            weights = (corr[mask] + 1) / 2
            edges = np.concatenate([edges, edges[[1, 0]]], axis=1)
            weights = np.concatenate([weights, weights])
            if weights.size:
                wmin, wmax = weights.min(), weights.max()
                if wmax > wmin:
                    weights = (weights - wmin) / (wmax - wmin)
    else:
        edges = np.zeros((2, 0), dtype=np.int64)
        weights = np.array([], dtype=np.float32)

    clustered_data = Data(
        x=x_new,
        edge_index=torch.tensor(edges, dtype=torch.long),
        edge_weight=torch.tensor(weights, dtype=torch.float),
        y=data.y.clone(),
        roi=roi_new
    )
    return clustered_data, mapping

def main():
    p = argparse.ArgumentParser(
        description="Cluster nodes of a PyG graph per ROI dynamically and save mapping."
    )
    p.add_argument("-i", "--input", required=True, help="Input .pt graph file.")
    p.add_argument("-o", "--output", required=True, help="Output clustered .pt file.")
    p.add_argument("--voxels_per_cluster", type=int, default=100,
                   help="Voxels per cluster for dynamic sizing (default=100).")
    p.add_argument("--min_clusters", type=int, default=1,
                   help="Minimum clusters per ROI (default=1).")
    p.add_argument("--max_clusters", type=int, default=50,
                   help="Maximum clusters per ROI (default=50).")
    p.add_argument("--corr_threshold", type=float, default=0.3,
                   help="Correlation threshold for edges (default=0.3).")
    p.add_argument("--max_distance", type=float, default=10.0,
                   help="Maximum spatial distance for edges (mm).")
    args = p.parse_args()

    data = torch.load(args.input)
    print(f"Loaded: {data.num_nodes} nodes, {len(torch.unique(data.roi))} ROIs, {data.num_edges} edges")

    clustered_data, mapping = cluster_graph_by_roi(
        data,
        voxels_per_cluster=args.voxels_per_cluster,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        corr_threshold=args.corr_threshold,
        max_distance=args.max_distance
    )
    print(f"Clustered: {clustered_data.num_nodes} nodes, {clustered_data.num_edges} edges")

    torch.save(clustered_data, args.output)
    print(f"Saved clustered graph to {args.output}")

    base, ext = os.path.splitext(args.output)
    map_path = f"{base}_mapping.json"
    with open(map_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved cluster-to-voxels mapping to {map_path}")

if __name__ == "__main__":
    main()