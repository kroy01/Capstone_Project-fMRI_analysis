import argparse
import torch
import numpy as np
from torch_geometric.data import Data

# Plotly imports:
import plotly.graph_objects as go
import plotly.io as pio


def visualize_fmri_graph_plotly(graph: Data, color_by='grayscale', sample_rate=1.0, out_html=None):
    """
    Visualize the 3D voxel graph in Plotly.

    Args:
        graph (Data): PyG Data object containing:
            graph.x -> node features, assumed to be [x, y, z, grayscale].
            graph.y -> graph label (optional).
        color_by (str): 'grayscale' or any future scheme for coloring.
        sample_rate (float): fraction of voxels to randomly sample (0, 1].
        out_html (str): optional path to save the interactive plot as HTML.

    Returns:
        fig (go.Figure): A Plotly Figure object.
    """
    # ---------------------------
    # 1. Extract node features
    # ---------------------------
    node_features = graph.x.cpu().numpy()
    coords = node_features[:, :3]  # (x, y, z)
    grayscale = node_features[:, 3]  # intensity

    # ---------------------------
    # 2. Optional downsampling
    # ---------------------------
    num_nodes = coords.shape[0]
    if sample_rate < 1.0:
        sample_size = int(num_nodes * sample_rate)
        idx = np.random.choice(num_nodes, sample_size, replace=False)
        coords = coords[idx]
        grayscale = grayscale[idx]

    # If you had multiple color schemes (e.g., label-based),
    # you'd adjust color array here. For now, it's intensity-based.
    color_values = grayscale

    # ---------------------------
    # 3. Create Plotly 3D scatter
    # ---------------------------
    scatter = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(
            size=2,  # marker size
            color=color_values,  # color by intensity
            colorscale='Viridis',  # or 'Hot', 'Cividis', etc.
            opacity=0.8,
            colorbar=dict(title='Grayscale')  # label colorbar
        )
    )

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title="fMRI Graph (Voxel-level 3D Visualization)",
    )

    # ---------------------------
    # 4. Optional: save or show
    # ---------------------------
    if out_html:
        pio.write_html(fig, file=out_html, auto_open=False)
        print(f"HTML visualization saved to: {out_html}")
    else:
        # Show figure in interactive browser window if possible
        fig.show()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize saved fMRI graph via Plotly")
    parser.add_argument("--path", type=str, required=True,
                        help="Path to the saved PyG Data .pt file")
    parser.add_argument("--sample_rate", type=float, default=1.0,
                        help="Downsampling rate (0,1], e.g. 0.1 for 10% of voxels")
    parser.add_argument("--out_html", type=str, default=None,
                        help="Path to save an interactive .html plot (optional)")
    args = parser.parse_args()

    # 1. Load the PyG Data object
    data = torch.load(args.path)
    if not isinstance(data, Data):
        raise ValueError("Loaded file is not a PyG Data object")

    # 2. Visualize
    visualize_fmri_graph_plotly(
        graph=data,
        sample_rate=args.sample_rate,
        out_html=args.out_html
    )