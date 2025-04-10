#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

app = dash.Dash(__name__)

# Global data references
NODE_COORDS = None      # shape (N, 3) for (x, y, z)
NODE_INTENSITY = None   # shape (N,) for coloring the scatter
BOLD_ARRAY = None       # shape (N, T) for each voxel's BOLD
COORD_TO_INDEX = None   # dict {(x, y, z): node_index}
EDGE_INDEX = None       # shape (2, E) numpy array holding edge connections
EDGE_WEIGHT = None      # numpy array holding edge weights for each edge
DEFAULT_SAMPLE_RATE = 0.1  # 10% downsampling by default for faster initial load


def create_3d_scatter_figure(coords, intensity, sample_rate=1.0):
    """
    Build a 3D scatter plot of voxel locations.
    """
    num_voxels = coords.shape[0]
    if sample_rate < 1.0:
        sample_size = int(num_voxels * sample_rate)
        idx = np.random.choice(num_voxels, sample_size, replace=False)
        coords_plot = coords[idx]
        intensity_plot = intensity[idx]
        indices_plot = idx
    else:
        coords_plot = coords
        intensity_plot = intensity
        indices_plot = np.arange(num_voxels)

    scatter = go.Scatter3d(
        x=coords_plot[:, 0],
        y=coords_plot[:, 1],
        z=coords_plot[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=intensity_plot,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='Intensity')
        ),
        text=[str(i) for i in indices_plot],
    )

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title="BOLD fMRI Graph (Voxel-level 3D)",
        clickmode='event+select'
    )
    return fig


def create_time_series_figure(time_series, voxel_index, coordinate):
    """
    Create a 2D line plot of a single voxel's BOLD time series.
    """
    coord_str = f"({int(coordinate[0])}, {int(coordinate[1])}, {int(coordinate[2])})"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(time_series)),
        y=time_series,
        mode='lines',
        name=f"Voxel {voxel_index} BOLD"
    ))
    fig.update_layout(
        title=f"BOLD Time Series - Voxel {voxel_index} {coord_str}",
        xaxis_title="Time (volume index)",
        yaxis_title="BOLD Intensity"
    )
    return fig


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
app.layout = html.Div([
    html.H1("BOLD fMRI Graph Visualization (Dash)"),

    # 3D Scatter Plot
    dcc.Graph(id='brain-3d-scatter'),

    # Time Series Plot
    dcc.Graph(id='time-series-plot'),

    # Controls for sample rate
    html.Div([
        html.Label("Voxel Sample Rate (0.01 to 1.0):"),
        dcc.Input(
            id='sample-rate-input',
            type='number',
            value=DEFAULT_SAMPLE_RATE,
            min=0.01,
            max=1.0,
            step=0.01
        ),
        html.Button("Update Plot", id="update-button"),
    ], style={'margin-bottom': '20px'}),

    # Hidden store for selected voxel
    dcc.Store(id='selected-voxel-store'),
    html.Button("Download CSV", id="download-button"),
    dcc.Download(id="download-dataframe-csv"),

    html.Hr(),

    # Input for manual coordinate selection
    html.H3("Manually Pick a (x,y,z) Coordinate:"),
    html.Div([
        "x: ", dcc.Input(id="x-input", type="number", value=0),
        " y: ", dcc.Input(id="y-input", type="number", value=0),
        " z: ", dcc.Input(id="z-input", type="number", value=0),
    ], style={'margin-bottom': '10px'}),
    html.Button("Find Voxel", id="find-voxel-button"),

    # New: Dynamic Table for Neighbour Voxels and Edge Weights
    html.Hr(),
    html.H3("Neighbour Voxels and Edge Weights:"),
    html.Div(id="neighbour-table-div")
])


# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------

@app.callback(
    Output('brain-3d-scatter', 'figure'),
    Input('update-button', 'n_clicks'),
    State('sample-rate-input', 'value')
)
def update_3d_figure(_, new_sample_rate):
    """
    Recreate the 3D scatter with the specified sample rate.
    """
    global NODE_COORDS, NODE_INTENSITY
    if NODE_COORDS is None or NODE_INTENSITY is None:
        return go.Figure()

    sample_rate = float(new_sample_rate) if new_sample_rate else 1.0
    return create_3d_scatter_figure(NODE_COORDS, NODE_INTENSITY, sample_rate=sample_rate)


@app.callback(
    Output('selected-voxel-store', 'data'),
    [
        Input('brain-3d-scatter', 'clickData'),
        Input('find-voxel-button', 'n_clicks')
    ],
    [
        State('x-input', 'value'),
        State('y-input', 'value'),
        State('z-input', 'value')
    ]
)
def store_voxel_index(clickData, find_n_clicks, x_val, y_val, z_val):
    """
    Set the selected voxel from either a 3D click or coordinate input.
    """
    from dash import callback_context as ctx

    if not ctx.triggered:
        return None

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'brain-3d-scatter':
        if not clickData or not clickData['points']:
            return None
        voxel_index_str = clickData['points'][0].get('text')
        if voxel_index_str is None:
            return None
        return int(voxel_index_str)

    elif triggered_id == 'find-voxel-button':
        coord_key = (int(x_val), int(y_val), int(z_val))
        voxel_idx = COORD_TO_INDEX.get(coord_key, None)
        return voxel_idx

    return None


@app.callback(
    Output('time-series-plot', 'figure'),
    Input('selected-voxel-store', 'data')
)
def update_time_series_plot(voxel_index):
    """
    Update the time series plot when the selected voxel changes.
    """
    global BOLD_ARRAY, NODE_COORDS
    empty_fig = go.Figure().update_layout(
        title="Click a voxel or pick a coordinate to see BOLD time series",
        xaxis_title="Time",
        yaxis_title="BOLD"
    )
    if voxel_index is None or BOLD_ARRAY is None:
        return empty_fig
    if voxel_index < 0 or voxel_index >= BOLD_ARRAY.shape[0]:
        return empty_fig

    time_series = BOLD_ARRAY[voxel_index]
    coordinate = NODE_COORDS[voxel_index]
    return create_time_series_figure(time_series, voxel_index, coordinate)


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-button", "n_clicks"),
    State("selected-voxel-store", "data"),
    prevent_initial_call=True
)
def download_csv(n_clicks, voxel_index):
    """
    Download a CSV of volume index vs. BOLD intensity for the selected voxel.
    """
    if voxel_index is None:
        raise dash.exceptions.PreventUpdate

    global BOLD_ARRAY, NODE_COORDS
    if BOLD_ARRAY is None:
        raise dash.exceptions.PreventUpdate
    if voxel_index < 0 or voxel_index >= BOLD_ARRAY.shape[0]:
        raise dash.exceptions.PreventUpdate

    ts = BOLD_ARRAY[voxel_index]
    coordinate = NODE_COORDS[voxel_index]
    df = pd.DataFrame({
        'volume_index': np.arange(len(ts)),
        'bold_intensity': ts
    })
    filename = f"voxel_{voxel_index}_{int(coordinate[0])}_{int(coordinate[1])}_{int(coordinate[2])}_bold.csv"
    return dcc.send_data_frame(df.to_csv, filename=filename, index=False)


@app.callback(
    Output("neighbour-table-div", "children"),
    Input("selected-voxel-store", "data")
)
def update_neighbour_table(voxel_index):
    """
    Update the neighbours table dynamically based on the selected voxel.
    For the chosen voxel, scan EDGE_INDEX to find all connected neighbours and 
    list their (x,y,z) coordinates along with the corresponding edge weight.
    Duplicate neighbour entries are removed.
    """
    global NODE_COORDS, EDGE_INDEX, EDGE_WEIGHT

    if voxel_index is None or EDGE_INDEX is None or EDGE_WEIGHT is None:
        return html.Div("Select a voxel to see its neighbours and edge weights.")

    # Use a dictionary to keep track of unique neighbours.
    neighbours = {}
    # EDGE_INDEX assumed shape: [2, E]
    for i in range(EDGE_INDEX.shape[1]):
        src = EDGE_INDEX[0, i]
        dst = EDGE_INDEX[1, i]
        if src == voxel_index or dst == voxel_index:
            neighbour_idx = dst if src == voxel_index else src
            # Convert coordinate to integer tuple.
            neighbour_coord = tuple(map(int, NODE_COORDS[neighbour_idx]))
            # Only add the neighbour if it's not already in the dictionary.
            if neighbour_coord not in neighbours:
                neighbours[neighbour_coord] = EDGE_WEIGHT[i]

    if not neighbours:
        return html.Div("No neighbouring voxels found for the selected voxel.")

    # Build an HTML table with a header and one row per unique neighbour
    header = [html.Tr([html.Th("Neighbour Voxel (x,y,z)"), html.Th("Edge Weight")])]
    rows = [html.Tr([html.Td(f"({n[0]}, {n[1]}, {n[2]})"), html.Td(f"{weight:.3f}")])
            for n, weight in neighbours.items()]
    table = html.Table(header + rows, style={'width': '50%', 'border': '1px solid black', 'borderCollapse': 'collapse'})

    return table


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dash-based fMRI GNN Visualizer with coordinate picking, CSV download, and neighbour table.")
    parser.add_argument("--path", type=str, required=True,
                        help="Path to the saved PyG Data .pt file (shape: [num_voxels, 3 + T]).")
    parser.add_argument("--port", type=int, default=8050, help="Port for the Dash server.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for the Dash server.")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode.")
    args = parser.parse_args()

    # Load the PyG Data object
    print("Loading the PyG Data ...")
    data = torch.load(args.path, weights_only=False)
    if not isinstance(data, Data):
        raise ValueError("Loaded file is not a PyG Data object.")

    node_features = data.x.cpu().numpy()  # shape: [N, 3 + T]
    if node_features.shape[1] < 4:
        raise ValueError("Expecting [x, y, z] + >=1 BOLD columns (total >=4 columns).")

    global NODE_COORDS, NODE_INTENSITY, BOLD_ARRAY, COORD_TO_INDEX, EDGE_INDEX, EDGE_WEIGHT
    N = node_features.shape[0]
    coords = node_features[:, :3]
    bold_data = node_features[:, 3:]

    # For coloring the 3D scatter, use the average BOLD if multiple timepoints
    if bold_data.shape[1] == 1:
        intensity = bold_data[:, 0]
    else:
        intensity = bold_data.mean(axis=1)

    NODE_COORDS = coords
    NODE_INTENSITY = intensity
    BOLD_ARRAY = bold_data

    # Build a dictionary for coordinate -> node index (assumes integer voxel indices)
    COORD_TO_INDEX = {}
    for idx in range(N):
        x, y, z = coords[idx].astype(int)
        COORD_TO_INDEX[(x, y, z)] = idx

    # Load edge information: edge_index and edge_weight.
    # Expecting edge_index of shape [2, E] and edge_weight with shape [E,]
    EDGE_INDEX = data.edge_index.cpu().numpy()
    EDGE_WEIGHT = data.edge_weight.numpy()

    print(f"Loaded graph with {N} voxels, each having {bold_data.shape[1]} timepoints.")
    print(f"Graph connectivity: {EDGE_INDEX.shape[1]} edges loaded.")
    print(f"Running Dash at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
