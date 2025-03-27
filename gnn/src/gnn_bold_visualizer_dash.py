import argparse
import numpy as np
import torch
from torch_geometric.data import Data

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

app = dash.Dash(__name__)

# Global variables (set after loading)
NODE_COORDS = None  # shape (N, 3)
NODE_INTENSITY = None  # shape (N,) - used for coloring the 3D scatter
BOLD_ARRAY = None  # shape (N, T) time series for each voxel
SAMPLE_RATE = 1.0


def create_3d_scatter_figure(coords, intensity, sample_rate=1.0):
    """
    Create a 3D scatter plot of voxel locations.

    Args:
        coords (ndarray): shape (N, 3) voxel coordinates [x, y, z].
        intensity (ndarray): shape (N,) for color (e.g., average BOLD).
        sample_rate (float): fraction of voxels to plot (downsampling for speed).

    Returns:
        go.Figure: 3D scatter plot with each voxel as a point.
    """
    num_voxels = coords.shape[0]
    if sample_rate < 1.0:
        sample_size = int(num_voxels * sample_rate)
        idx = np.random.choice(num_voxels, sample_size, replace=False)
        coords_sampled = coords[idx]
        intensity_sampled = intensity[idx]
        indices_for_hover = idx
    else:
        coords_sampled = coords
        intensity_sampled = intensity
        indices_for_hover = np.arange(num_voxels)

    # We store the voxel index in 'text' so we can retrieve it in the callback
    scatter = go.Scatter3d(
        x=coords_sampled[:, 0],
        y=coords_sampled[:, 1],
        z=coords_sampled[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=intensity_sampled,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='Intensity')
        ),
        text=[str(i) for i in indices_for_hover],  # use text to store indices
        # customdata=indices_for_hover could also be used if your Plotly version supports it in clickData
    )

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title="BOLD fMRI Graph (Voxel-level 3D)",
        clickmode='event+select'  # enable click data
    )
    return fig


def create_time_series_figure(time_series, voxel_index):
    """
    Create a 2D plot of the BOLD time series for a single voxel.

    Args:
        time_series (ndarray): shape (T,) BOLD values across time.
        voxel_index (int): index of the voxel for labeling.

    Returns:
        go.Figure: 2D line plot of BOLD vs. volume index.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(time_series)),
        y=time_series,
        mode='lines',
        name=f'Voxel {voxel_index} BOLD'
    ))
    fig.update_layout(
        title=f"BOLD Time Series - Voxel {voxel_index}",
        xaxis_title="Time (volume index)",
        yaxis_title="BOLD Intensity"
    )
    return fig


app.layout = html.Div([
    html.H1("BOLD fMRI Graph Visualization (Dash)"),

    dcc.Graph(id='brain-3d-scatter'),
    dcc.Graph(id='time-series-plot'),

    html.Label("Voxel Sample Rate (0.01 to 1.0):"),
    dcc.Input(
        id='sample-rate-input',
        type='number',
        value=1.0,
        min=0.01,
        max=1.0,
        step=0.01
    ),
    html.Button("Update Plot", id="update-button"),
])


@app.callback(
    Output('brain-3d-scatter', 'figure'),
    Input('update-button', 'n_clicks'),
    Input('sample-rate-input', 'value')
)
def update_3d_figure(_, new_sample_rate):
    """
    Recreate the 3D scatter with a new sample rate if user adjusts it.
    """
    global NODE_COORDS, NODE_INTENSITY
    if NODE_COORDS is None or NODE_INTENSITY is None:
        return go.Figure()
    sample_rate = float(new_sample_rate) if new_sample_rate else 1.0
    fig = create_3d_scatter_figure(NODE_COORDS, NODE_INTENSITY, sample_rate=sample_rate)
    return fig


@app.callback(
    Output('time-series-plot', 'figure'),
    Input('brain-3d-scatter', 'clickData')
)
def display_click_voxel_time_series(clickData):
    """
    When user clicks on a voxel in the 3D scatter, show its time series in a 2D plot.
    """
    global BOLD_ARRAY

    # A default empty figure if no click event
    empty_fig = go.Figure().update_layout(
        title="Click a voxel to see its BOLD time series",
        xaxis_title="Time",
        yaxis_title="BOLD"
    )
    if clickData is None or BOLD_ARRAY is None:
        return empty_fig

    # Retrieve index from 'text' property
    point_data = clickData['points'][0]
    # 'text' is stored as a string index => convert to int
    voxel_index = int(point_data['text'])

    if voxel_index < 0 or voxel_index >= BOLD_ARRAY.shape[0]:
        return empty_fig

    time_series = BOLD_ARRAY[voxel_index]
    fig = create_time_series_figure(time_series, voxel_index)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Dash-based visualizer for a BOLD-graph PyG Data.")
    parser.add_argument("--path", type=str, required=True,
                        help="Path to the saved PyG Data .pt file (with x=[num_voxels, 3 + T]).")
    parser.add_argument("--port", type=int, default=8050, help="Port for the Dash server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for the Dash server.")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode.")
    args = parser.parse_args()

    # 1) Load the PyG Data object
    data = torch.load(args.path)
    if not isinstance(data, Data):
        raise ValueError("Loaded file is not a PyG Data object")

    # data.x shape = [N, 3 + T] => first 3 columns are [x, y, z], next T are BOLD
    node_features = data.x.cpu().numpy()
    if node_features.shape[1] < 4:
        raise ValueError("We expect at least 4 columns: 3 for coords + >=1 for BOLD timepoints")

    global NODE_COORDS, NODE_INTENSITY, BOLD_ARRAY

    # coords = first 3 columns
    coords = node_features[:, :3]
    # BOLD = columns [3:]
    time_series = node_features[:, 3:]

    # We'll color by average BOLD if multiple timepoints:
    # or if only one column, we color by that single intensity
    if time_series.shape[1] == 1:
        intensity = time_series[:, 0]
    else:
        intensity = time_series.mean(axis=1)

    NODE_COORDS = coords
    NODE_INTENSITY = intensity
    BOLD_ARRAY = time_series  # shape = [N, T]

    print(
        f"Loaded data with {node_features.shape[0]} voxels, each has {node_features.shape[1]} features (3 coords + {time_series.shape[1]} timepoints).")

    print(f"Starting Dash server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
