#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from torch_geometric.data import Data

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

app = dash.Dash(__name__)

# Global variables (populated after loading data)
NODE_COORDS = None       # shape (N, 3), float or int voxel coords
NODE_INTENSITY = None    # shape (N,) - for coloring the 3D scatter
BOLD_ARRAY = None        # shape (N, T) time series for each voxel
VOXEL_DICT = {}          # maps (x, y, z) -> index in [0..N-1]
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
        text=[str(i) for i in indices_for_hover],  # store original index as text
    )

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title="BOLD fMRI Graph (Voxel-level 3D)",
        clickmode='event+select'  # enable click events
    )
    return fig


def create_time_series_figure(time_series, voxel_label="Voxel"):
    """
    Create a 2D plot of the BOLD time series for a single voxel.

    Args:
        time_series (ndarray): shape (T,) BOLD values across time.
        voxel_label (str): name/label for the voxel in the title.

    Returns:
        go.Figure: 2D line plot of BOLD vs. time index.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(time_series)),
        y=time_series,
        mode='lines',
        name=f'{voxel_label} BOLD'
    ))
    fig.update_layout(
        title=f"BOLD Time Series - {voxel_label}",
        xaxis_title="Time (volume index)",
        yaxis_title="BOLD Intensity"
    )
    return fig


# Layout with:
# 1) A 3D scatter (fMRI brain)
# 2) The time-series plot
# 3) Input fields (x, y, z) + button to query
# 4) A sample rate input
app.layout = html.Div([
    html.H1("BOLD fMRI Graph Visualization (Dash)"),

    dcc.Graph(id='brain-3d-scatter'),
    dcc.Graph(id='time-series-plot'),

    html.Br(),

    html.Div([
        html.Label("Voxel Sample Rate (0.01 to 1.0):"),
        dcc.Input(
            id='sample-rate-input',
            type='number',
            value=1.0,
            min=0.01,
            max=1.0,
            step=0.01
        ),
        html.Button("Update 3D Plot", id="update-plot-button"),
    ]),

    html.Hr(),
    html.Label("Or enter voxel coordinates (x, y, z):"),

    html.Div([
        dcc.Input(id='voxel-x-input', type='number', placeholder="X coord"),
        dcc.Input(id='voxel-y-input', type='number', placeholder="Y coord"),
        dcc.Input(id='voxel-z-input', type='number', placeholder="Z coord"),
        html.Button("Show Time Series", id="voxel-query-button"),
    ], style={'marginTop': '10px'}),

], style={'width': '80%', 'margin': 'auto'})


@app.callback(
    Output('brain-3d-scatter', 'figure'),
    Input('update-plot-button', 'n_clicks'),
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
    # We have two possible triggers:
    # 1) A click on the 3D scatter
    # 2) The "Show Time Series" button
    #
    # We'll handle both in a single callback by examining which one triggered.
    [
        Input('brain-3d-scatter', 'clickData'),
        Input('voxel-query-button', 'n_clicks'),
    ],
    [
        State('voxel-x-input', 'value'),
        State('voxel-y-input', 'value'),
        State('voxel-z-input', 'value'),
    ],
    prevent_initial_call=True
)
def display_click_or_coord(clickData, _, vx, vy, vz):
    """
    If user clicks on a voxel, show that voxel's time series.
    If user enters coordinates and clicks "Show Time Series", show that voxel's time series.
    """
    global BOLD_ARRAY, VOXEL_DICT

    # Default figure if no valid input
    empty_fig = go.Figure().update_layout(
        title="Click a voxel or enter coords to see its BOLD time series",
        xaxis_title="Time",
        yaxis_title="BOLD"
    )

    # Check which input fired using dash.callback_context
    ctx = dash.callback_context
    if not ctx.triggered:
        return empty_fig

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # 1) If triggered by clicking on 3D scatter
    if trigger_id == 'brain-3d-scatter' and clickData is not None:
        # get the integer voxel index from the 'text' property
        point_data = clickData['points'][0]
        voxel_index = int(point_data['text'])

        if voxel_index < 0 or voxel_index >= BOLD_ARRAY.shape[0]:
            return empty_fig

        time_series = BOLD_ARRAY[voxel_index]
        return create_time_series_figure(time_series, voxel_label=f"Voxel {voxel_index}")

    # 2) If triggered by the "Show Time Series" button
    elif trigger_id == 'voxel-query-button':
        # The user typed (vx, vy, vz)
        if vx is None or vy is None or vz is None:
            # missing coordinate
            fig = go.Figure().update_layout(
                title="Please enter valid voxel coordinates (x, y, z)."
            )
            return fig

        # convert to int if your coords are integer-based
        # if your data had float coords, you may need a rounding or nearest lookup
        try:
            vx, vy, vz = int(vx), int(vy), int(vz)
        except ValueError:
            return go.Figure().update_layout(title="Coordinates must be integer or near-integer.")

        key = (vx, vy, vz)
        if key not in VOXEL_DICT:
            fig = go.Figure().update_layout(title=f"No voxel found at {key}")
            return fig

        voxel_index = VOXEL_DICT[key]
        time_series = BOLD_ARRAY[voxel_index]
        return create_time_series_figure(time_series, voxel_label=f"Coord {key}")

    # If neither case applies, just return empty
    return empty_fig


def main():
    parser = argparse.ArgumentParser(description="Dash-based visualizer for a BOLD-graph PyG Data, with coordinate lookup.")
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

    # data.x shape = [N, 3 + T] => first 3 columns: [x, y, z], next T columns: BOLD
    node_features = data.x.cpu().numpy()  # shape (N, 3+T)
    if node_features.shape[1] < 4:
        raise ValueError("We expect at least 4 columns: 3 for coords + >=1 for BOLD timepoints")

    global NODE_COORDS, NODE_INTENSITY, BOLD_ARRAY, VOXEL_DICT

    # 2) Extract coords and BOLD
    coords = node_features[:, :3]       # shape (N, 3)
    time_series = node_features[:, 3:]  # shape (N, T)

    # We'll color by average BOLD if multiple timepoints; or single if T=1
    if time_series.shape[1] == 1:
        intensity = time_series[:, 0]
    else:
        intensity = time_series.mean(axis=1)

    NODE_COORDS = coords
    NODE_INTENSITY = intensity
    BOLD_ARRAY = time_series

    # 3) Build a dict to map (x, y, z) -> index
    #    (Assuming coords are integer-based. If float, you may need nearest match.)
    VOXEL_DICT = {}
    for i in range(coords.shape[0]):
        vx, vy, vz = coords[i]
        # Convert to int if your data is integer-based
        # If your data is truly float, consider rounding or a nearest neighbor approach
        key = (int(vx), int(vy), int(vz))
        VOXEL_DICT[key] = i

    print(f"Loaded data with {coords.shape[0]} voxels, each has {time_series.shape[1]} timepoints.")
    print(f"Starting Dash server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
