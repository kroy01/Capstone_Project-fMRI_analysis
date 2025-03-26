# import nibabel as nib
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.decomposition import PCA
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader  # Updated import to suppress deprecation warning
# from torch_geometric.nn import GCNConv, global_mean_pool
#
#
# def load_fmri_graph(nii_path, label, pca_components=20, connectivity=6):
#     """Load fMRI from NIfTI and construct graph with spatial connectivity"""
#     # Load 4D fMRI data
#     img = nib.load(nii_path)
#     data = img.get_fdata()  # Shape: (X, Y, Z, Time)
#     x_dim, y_dim, z_dim, t_dim = data.shape
#
#     # Reshape to voxels × time
#     data_2d = data.reshape(-1, t_dim)
#
#     # Create brain mask (non-flat time series)
#     mask = np.std(data_2d, axis=1) > 1e-6
#     coords = np.argwhere(mask.reshape(x_dim, y_dim, z_dim))
#     features = data_2d[mask]
#
#     # Dimensionality reduction per voxel
#     pca = PCA(n_components=pca_components)
#     features = pca.fit_transform(features)
#
#     # Create coordinate to index mapping
#     coord_dict = {(x, y, z): i for i, (x, y, z) in enumerate(coords)}
#
#     # Generate spatial edges (6/26 connectivity)
#     edges = []
#     deltas = []
#     if connectivity == 6:
#         deltas = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
#     elif connectivity == 26:
#         deltas = [(dx, dy, dz) for dx in [-1, 0, 1]
#                   for dy in [-1, 0, 1]
#                   for dz in [-1, 0, 1] if not (dx == 0 and dy == 0 and dz == 0)]
#
#     for i, (x, y, z) in enumerate(coords):
#         for dx, dy, dz in deltas:
#             nx, ny, nz = x + dx, y + dy, z + dz
#             if (nx, ny, nz) in coord_dict:
#                 edges.append((i, coord_dict[(nx, ny, nz)]))
#
#     edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#
#     return Data(
#         x=torch.tensor(features, dtype=torch.float),
#         edge_index=edge_index,
#         y=torch.tensor([label], dtype=torch.long)
#     )
#
#
# class fMRI_GNN(nn.Module):
#     """GNN for graph-level fMRI classification"""
#
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.conv3 = GCNConv(hidden_dim, hidden_dim)
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(hidden_dim // 2, output_dim)
#         )
#
#     def forward(self, data):
#         # Graph convolution
#         x = F.relu(self.conv1(data.x, data.edge_index))
#         x = F.relu(self.conv2(x, data.edge_index))
#         x = self.conv3(x, data.edge_index)
#
#         # Global pooling and classification
#         x = global_mean_pool(x, data.batch)
#         return self.classifier(x)
#
#
# # Example Usage
# if __name__ == "__main__":
#     # Parameters
#     BATCH_SIZE = 2  # Adjusted to match the number of samples
#     PCA_COMPONENTS = 20
#     HIDDEN_DIM = 128
#
#     # 1. Load dataset
#     nii_files = [
#         "../resources/filtered_func_data.nii.gz"
#     ]
#     labels = [0, 1]  # Ensure this matches the number of files
#
#     dataset = [load_fmri_graph(f, l, PCA_COMPONENTS) for f, l in zip(nii_files, labels)]
#     loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
#
#     # 2. Initialize model
#     model = fMRI_GNN(
#         input_dim=PCA_COMPONENTS,
#         hidden_dim=HIDDEN_DIM,
#         output_dim=2  # Number of classes
#     )
#
#     # 3. Training setup
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()
#
#     # 4. Training loop
#     for epoch in range(10):
#         for batch in loader:
#             print(f"Batch y shape: {batch.y.shape}, Number of graphs: {batch.num_graphs}")
#             optimizer.zero_grad()
#             out = model(batch)
#             print(f"Model output shape: {out.shape}")
#             loss = criterion(out, batch.y)  # No .squeeze() needed
#             loss.backward()
#             optimizer.step()
#         print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

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
i=0
for batch in loader:
    print("Node feature shape:", batch.x.shape)
    print("Edge index shape:", batch.edge_index.shape)
    print("Graph labels:", batch.y)
    # Pass to your GNN...
    torch.save(batch, f'../resources/subject_{i}_graph.pt')
    i+=1