import nibabel as nib
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_fmri_graph(
    nii_path,
    label,
    connectivity=6,
    threshold=1e-6
):
    """
    Load 3D (or time-averaged 4D) fMRI from NIfTI and construct a graph.
      - Each voxel becomes a node with features = (x, y, z, grayscale).
      - Edges represent spatial connectivity (6- or 26-neighborhood).
      - Edges are made bidirectional.
    """
    # -------------------------------------------------------------------------
    # 1. Load NIfTI
    #    If you have a 4D fMRI, we can take the mean across time to get a single
    #    grayscale value per voxel. Alternatively, if your NIfTI is already 3D
    #    (e.g., T1 or time-averaged fMRI), just read it directly.
    # -------------------------------------------------------------------------
    img = nib.load(nii_path)
    data_4d = img.get_fdata()  # shape could be (X, Y, Z, T) or (X, Y, Z)

    if data_4d.ndim == 4:
        # Average across time => shape (X, Y, Z)
        data_3d = np.mean(data_4d, axis=3)
    else:
        # Already 3D
        data_3d = data_4d

    x_dim, y_dim, z_dim = data_3d.shape

    # -------------------------------------------------------------------------
    # 2. (Optional) Mask out voxels near zero. If you want all voxels, skip.
    # -------------------------------------------------------------------------
    mask = data_3d > threshold  # simple threshold-based mask
    coords = np.argwhere(mask)
    # E.g. coords: list of (x, y, z) for non-zero voxels

    # -------------------------------------------------------------------------
    # 3. Build node features: (x, y, z, grayscale)
    # -------------------------------------------------------------------------
    # Flatten the data volume
    flat_data = data_3d.reshape(-1)

    node_features = []
    for (x, y, z) in coords:
        grayscale_val = flat_data[x * y_dim * z_dim + y * z_dim + z]
        node_features.append([float(x), float(y), float(z), float(grayscale_val)])
    node_features = np.array(node_features, dtype=np.float32)

    # -------------------------------------------------------------------------
    # 4. Build edges based on connectivity (6- or 26-neighborhood).
    #    We store edges in both directions to ensure bidirectionality.
    # -------------------------------------------------------------------------
    # Map each (x, y, z) to an index i in node_features
    coord_dict = {(x, y, z): i for i, (x, y, z) in enumerate(coords)}

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
        raise ValueError("connectivity must be 6 or 26")

    edges = []
    for (x, y, z) in coords:
        this_index = coord_dict[(x, y, z)]
        for (dx, dy, dz) in neighbor_deltas:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (nx, ny, nz) in coord_dict:
                neighbor_index = coord_dict[(nx, ny, nz)]
                edges.append((this_index, neighbor_index))
                edges.append((neighbor_index, this_index))  # Make bidirectional

    # -------------------------------------------------------------------------
    # 5. Convert to torch tensors
    # -------------------------------------------------------------------------
    x_tensor = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Single-label for the entire 3D volume
    y_tensor = torch.tensor([label], dtype=torch.long)

    # -------------------------------------------------------------------------
    # 6. Return PyG Data object
    # -------------------------------------------------------------------------
    return Data(
        x=x_tensor,
        edge_index=edge_index,
        y=y_tensor
    )



# Example usage
nii_files = [
    "../resources/filtered_func_data.nii.gz"
]
labels = [0]  # 1 class(MCI), more can be added

# Build dataset
dataset = []
for f, lbl in zip(nii_files, labels):
    g_data = load_fmri_graph(f, label=lbl, connectivity=6)
    dataset.append(g_data)

# Create DataLoader
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Now each graph has:
#   x.shape = [num_nodes, 4]
#   edge_index.shape = [2, num_edges]
#   y.shape = [1]
i=1
for batch in loader:
    print("Node feature shape:", batch.x.shape)
    print("Edge index shape:", batch.edge_index.shape)
    print("Graph labels:", batch.y)
    # Pass to your GNN...
    torch.save(batch, f'../resources/subject_{i}_graph.pt')
    i+=1