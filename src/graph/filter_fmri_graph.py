#!/usr/bin/env python3
import argparse
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

def filter_graph_by_roi(data: Data, roi_ids: list[int]) -> Data:
    """
    Return a new Data object containing only nodes whose data.roi is in roi_ids,
    with edges (and edge_weight) restricted to those nodes.
    """
    # build mask of nodes to keep
    roi_tensor = torch.tensor(roi_ids, dtype=data.roi.dtype, device=data.roi.device)
    node_mask = torch.isin(data.roi, roi_tensor)

    # get indices of kept nodes
    kept_idx = node_mask.nonzero(as_tuple=False).view(-1)

    # extract subgraph (relabels nodes to 0..N-1, filters edge_weight)
    edge_index, edge_weight = subgraph(
        kept_idx,
        data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
        edge_attr=data.edge_weight
    )

    # slice node features and roi labels
    x = data.x[kept_idx]
    roi = data.roi[kept_idx]
    # graph-level label stays the same
    y = data.y

    return Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        roi=roi
    )

def parse_roi_list(s: str) -> list[int]:
    """
    Parse a comma‑separated list of ints, e.g. "17,18,53,54"
    """
    return [int(tok) for tok in s.split(',') if tok.strip()]

def main():
    p = argparse.ArgumentParser(
        description="Load a PyG .pt graph, filter by ROI IDs, and save the pruned graph."
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input .pt file (a torch_geometric Data object)."
    )
    p.add_argument(
        "--output", "-o",
        required=True,
        help="Where to save the filtered .pt graph."
    )
    p.add_argument(
        "--rois", "-r",
        required=True,
        help="Comma-separated ROI IDs to keep, e.g. '17,18,53,54,1006'"
    )
    args = p.parse_args()

    # load
    data: Data = torch.load(args.input)
    print(f"Loaded graph: {data.num_nodes} nodes, {data.num_edges} edges, {len(torch.unique(data.roi))} unique ROIs")

    # filter
    roi_list = parse_roi_list(args.rois)
    filtered = filter_graph_by_roi(data, roi_list)
    print(f"After filtering: {filtered.num_nodes} nodes, {filtered.num_edges} edges, {len(torch.unique(filtered.roi))} unique ROIs")

    # save
    torch.save(filtered, args.output)
    print(f"Saved filtered graph to {args.output}")

if __name__ == "__main__":
    main()
