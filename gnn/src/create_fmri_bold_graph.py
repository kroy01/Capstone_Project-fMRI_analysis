import argparse
import nibabel as nib
import torch
import numpy as np
from torch_geometric.data import Data

def load_fmri_graph_bold_with_coords(
        nii_path,
        label=0,
        connectivity=6,
        threshold=1e-6
):
    """
    Load a 4D fMRI from NIfTI and construct a PyG Data graph, storing:
      - x[:, :3] = (x, y, z) voxel coordinates
      - x[:, 3:] = BOLD time series (T timepoints)
      - edge_index for 6- or 26-neighborhood (bidirectional)
      - y = single integer label for the entire 4D scan

    Args:
        nii_path (str): Path to the 4D fMRI NIfTI file.
        label (int): Graph-level label for classification/regression (default=0).
        connectivity (int): 6 or 26 connectivity (default=6).
        threshold (float): Mask threshold to remove near-zero voxels (default=1e-6).

    Returns:
        Data: A torch_geometric.data.Data object with:
              x: [num_voxels, 3 + T]
                 - first 3 columns: [x, y, z]
                 - next T columns: the BOLD time series
              edge_index: [2, num_edges]
              y: [1] (the label)
    """
    # --------------------------
    # 1. Load the fMRI (4D)
    # --------------------------
    img = nib.load(nii_path)
    data_4d = img.get_fdata()  # shape: (X, Y, Z, T)
    x_dim, y_dim, z_dim, t_dim = data_4d.shape

    # --------------------------
    # 2. Brain Mask
    #    Simple threshold on the mean across time
    # --------------------------
    mean_3d = data_4d.mean(axis=3)  # shape: (X, Y, Z)
    mask = mean_3d > threshold
    coords = np.argwhere(mask)  # list of voxel coordinates (x, y, z)

    # --------------------------
    # 3. Build Node Features:
    #    x,y,z + entire time series -> shape (3 + T,)
    # --------------------------
    node_features = []
    for (vx, vy, vz) in coords:
        # Extract the time series for the given voxel
        time_series = data_4d[vx, vy, vz, :]  # shape: (T,)
        # Combine spatial coordinates and the time series:
        combined = np.concatenate(([vx, vy, vz], time_series))
        node_features.append(combined)

    # Convert to numpy array: shape [num_voxels, 3 + T]
    node_features = np.array(node_features, dtype=np.float32)

    # --------------------------
    # 4. Build Edges (Spatial Connectivity)
    #    Bidirectional connectivity based on 6- or 26-neighborhood
    # --------------------------
    coord_dict = {(vx, vy, vz): i for i, (vx, vy, vz) in enumerate(coords)}

    if connectivity == 6:
        neighbor_deltas = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1)
        ]
    elif connectivity == 26:
        neighbor_deltas = [
            (dx, dy, dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
    else:
        raise ValueError("Connectivity must be 6 or 26")

    edges = []
    for (vx, vy, vz) in coords:
        this_index = coord_dict[(vx, vy, vz)]
        for (dx, dy, dz) in neighbor_deltas:
            nx, ny, nz = vx + dx, vy + dy, vz + dz
            if (nx, ny, nz) in coord_dict:
                neighbor_index = coord_dict[(nx, ny, nz)]
                edges.append((this_index, neighbor_index))
                # Since the neighbor relationship is bidirectional,
                # the reverse edge is added (this may be redundant if already added)
                edges.append((neighbor_index, this_index))
                
    # --------------------------
    # 5. Compute Edge Weights based on fMRI time series correlations
    # --------------------------
    edge_weights = []
    # Here, the node_features array has columns [x, y, z, ...time series...]
    for (i, j) in edges:
        ts_i = node_features[i][3:]  # Get BOLD time series for voxel i
        ts_j = node_features[j][3:]  # Get BOLD time series for voxel j
        
        # Compute Pearson correlation coefficient between the two time series
        # (np.corrcoef returns a 2x2 matrix)
        r = np.corrcoef(ts_i, ts_j)[0, 1]
        
        # Normalize the correlation value from [-1, 1] to [0, 1]
        norm_corr = (r + 1) / 2.0

        # Assign weight: base value 1 plus normalized correlation
        raw_weight = 1 + norm_corr

        edge_weights.append(raw_weight)
    
    edge_weights = np.array(edge_weights, dtype=np.float32)
    
    # Normalize final edge weights to be between 0 and 1 using min–max normalization
    min_w = edge_weights.min()
    max_w = edge_weights.max()
    if max_w - min_w > 0:
        final_weights = (edge_weights - min_w) / (max_w - min_w)
    else:
        # In the unlikely event all weights are identical, use zeros
        final_weights = np.zeros_like(edge_weights)
    
    # --------------------------
    # 6. Wrap Data in PyTorch Tensors and Return PyG Data
    # --------------------------
    x_tensor = torch.tensor(node_features, dtype=torch.float)  # Node features tensor: [num_voxels, 3+T]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Edge index: [2, num_edges]
    y_tensor = torch.tensor([label], dtype=torch.long)  # Graph label
    
    # Create an edge attribute tensor (edge weights)
    edge_weight_tensor = torch.tensor(final_weights, dtype=torch.float)

    # Return the data object with nodes, edges, and edge weights (you can add edge_weight as an attribute)
    return Data(x=x_tensor, edge_index=edge_index, y=y_tensor, edge_weight=edge_weight_tensor)

# Example usage:
# data = load_fmri_graph_with_weights("path_to_4d_fmri.nii", label=1, connectivity=6)


def main():
    parser = argparse.ArgumentParser(description="Create a GNN-ready graph from 4D fMRI (coords + full BOLD).")
    parser.add_argument("--nii_path", type=str, required=True,
                        help="Path to the 4D fMRI NIfTI file.")
    parser.add_argument("--label", type=int, default=0,
                        help="Integer label for the entire 4D scan (default=0).")
    parser.add_argument("--connectivity", type=int, default=6,
                        choices=[6, 26],
                        help="Neighborhood connectivity: 6 or 26 (default=6).")
    parser.add_argument("--threshold", type=float, default=1e-6,
                        help="Threshold for masking near-zero voxels (default=1e-6).")
    parser.add_argument("--out_path", type=str, default="bold_graph.pt",
                        help="Output file path for the saved PyG Data object.")
    args = parser.parse_args()

    # 1. Load the 4D fMRI graph with (x,y,z) + full BOLD timeseries per voxel
    graph_data = load_fmri_graph_bold_with_coords(
        nii_path=args.nii_path,
        label=args.label,
        connectivity=args.connectivity,
        threshold=args.threshold
    )

    # 2. Print shapes/info
    print(f"Number of nodes        : {graph_data.num_nodes}")
    print(f"Node feature shape     : {graph_data.x.shape}")  # [num_voxels, 3 + T]
    print(f"Number of edges        : {graph_data.num_edges}")
    print(f"Graph label (y)        : {graph_data.y.tolist()}")

    # 3. Save
    torch.save(graph_data, args.out_path)
    print(f"Graph saved to: {args.out_path}")


if __name__ == "__main__":
    main()
