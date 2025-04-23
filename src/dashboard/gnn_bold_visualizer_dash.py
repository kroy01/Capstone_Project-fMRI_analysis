# # import argparse
# # import numpy as np
# # import pandas as pd
# # import torch
# # from torch_geometric.data import Data
# #
# # import dash
# # from dash import dcc, html, dash_table
# # from dash.dependencies import Input, Output, State
# # import plotly.graph_objs as go
# #
# # app = dash.Dash(__name__)
# #
# # # Global data references
# # NODE_COORDS = None      # shape (N, 3) for (x, y, z)
# # NODE_INTENSITY = None   # shape (N,) for coloring the scatter
# # BOLD_ARRAY = None       # shape (N, T) for each voxel's BOLD
# # COORD_TO_INDEX = None   # dict {(x, y, z): node_index}
# # EDGE_INDEX = None       # shape (2, E) numpy array holding edge connections
# # EDGE_WEIGHT = None      # numpy array holding edge weights for each edge
# # DEFAULT_SAMPLE_RATE = 0.1  # 10% downsampling by default for faster initial load
# #
# #
# # def create_3d_scatter_figure(coords, intensity, sample_rate=1.0):
# #     """
# #     Build a 3D scatter plot of voxel locations.
# #     """
# #     num_voxels = coords.shape[0]
# #     if sample_rate < 1.0:
# #         sample_size = int(num_voxels * sample_rate)
# #         idx = np.random.choice(num_voxels, sample_size, replace=False)
# #         coords_plot = coords[idx]
# #         intensity_plot = intensity[idx]
# #         indices_plot = idx
# #     else:
# #         coords_plot = coords
# #         intensity_plot = intensity
# #         indices_plot = np.arange(num_voxels)
# #
# #     scatter = go.Scatter3d(
# #         x=coords_plot[:, 0],
# #         y=coords_plot[:, 1],
# #         z=coords_plot[:, 2],
# #         mode='markers',
# #         marker=dict(
# #             size=3,
# #             color=intensity_plot,
# #             colorscale='Viridis',
# #             opacity=0.8,
# #             colorbar=dict(title='Intensity')
# #         ),
# #         text=[str(i) for i in indices_plot],
# #     )
# #
# #     fig = go.Figure(data=[scatter])
# #     fig.update_layout(
# #         scene=dict(
# #             xaxis_title='X',
# #             yaxis_title='Y',
# #             zaxis_title='Z'
# #         ),
# #         title="BOLD fMRI Graph (Voxel-level 3D)",
# #         clickmode='event+select'
# #     )
# #     return fig
# #
# #
# # def create_time_series_figure(time_series, voxel_index, coordinate):
# #     """
# #     Create a 2D line plot of a single voxel's BOLD time series.
# #     """
# #     coord_str = f"({int(coordinate[0])}, {int(coordinate[1])}, {int(coordinate[2])})"
# #     fig = go.Figure()
# #     fig.add_trace(go.Scatter(
# #         x=np.arange(len(time_series)),
# #         y=time_series,
# #         mode='lines',
# #         name=f"Voxel {voxel_index} BOLD"
# #     ))
# #     fig.update_layout(
# #         title=f"BOLD Time Series - Voxel {voxel_index} {coord_str}",
# #         xaxis_title="Time (volume index)",
# #         yaxis_title="BOLD Intensity"
# #     )
# #     return fig
# #
# #
# # # ----------------------------------------------------------------------------
# # # Layout
# # # ----------------------------------------------------------------------------
# # app.layout = html.Div([
# #     # Top heading (centered)
# #     html.H1("BOLD fMRI Graph Visualization (Dash)", style={'textAlign': 'center'}),
# #
# #     # Row container with flex for left and right components
# #     html.Div([
# #         # Left section (3D Scatter Plot) ~66% width
# #         html.Div([
# #             dcc.Graph(id='brain-3d-scatter'),
# #         ], style={'flex': '2', 'padding': '10px'}),
# #
# #         # Right section (controls and tables) ~33% width
# #         html.Div([
# #             html.H3("Neighbour Voxels and Edge Weights:"),
# #             html.Div(id="neighbour-table-div", style={'margin-bottom': '20px'}),
# #
# #             html.H3("Manually Pick a (x,y,z) Coordinate:"),
# #             html.Div([
# #                 "x: ", dcc.Input(id="x-input", type="number", value=0, style={'width': '15%'}),
# #                 " y: ", dcc.Input(id="y-input", type="number", value=0, style={'width': '15%'}),
# #                 " z: ", dcc.Input(id="z-input", type="number", value=0, style={'width': '15%'}),
# #             ], style={'margin-bottom': '10px'}),
# #             html.Button("Find Voxel", id="find-voxel-button", style={'margin-bottom': '20px'}),
# #
# #             html.Div([
# #                 html.Label("Voxel Sample Rate (0.01 to 1.0):"),
# #                 dcc.Input(
# #                     id='sample-rate-input',
# #                     type='number',
# #                     value=DEFAULT_SAMPLE_RATE,
# #                     min=0.01,
# #                     max=1.0,
# #                     step=0.01,
# #                     style={'margin-left': '10px', 'margin-right': '10px'}
# #                 ),
# #                 html.Button("Update Plot", id="update-button"),
# #             ], style={'margin-bottom': '20px'}),
# #
# #             html.Button("Download CSV", id="download-button"),
# #             dcc.Download(id="download-dataframe-csv"),
# #
# #         ], style={'flex': '1', 'padding': '10px'})
# #     ], style={'display': 'flex', 'width': '100%'}),
# #
# #     html.Hr(),
# #
# #     # Bottom: Time Series Plot
# #     html.H3("Selected Voxel's Time Series Plot (Interactive)", style={'textAlign': 'center'}),
# #     dcc.Graph(id='time-series-plot'),
# #
# #     # Hidden store for selected voxel
# #     dcc.Store(id='selected-voxel-store'),
# # ])
# #
# #
# # # ----------------------------------------------------------------------------
# # # Callbacks
# # # ----------------------------------------------------------------------------
# #
# # @app.callback(
# #     Output('brain-3d-scatter', 'figure'),
# #     Input('update-button', 'n_clicks'),
# #     State('sample-rate-input', 'value')
# # )
# # def update_3d_figure(_, new_sample_rate):
# #     """
# #     Recreate the 3D scatter with the specified sample rate.
# #     """
# #     global NODE_COORDS, NODE_INTENSITY
# #     if NODE_COORDS is None or NODE_INTENSITY is None:
# #         return go.Figure()
# #
# #     sample_rate = float(new_sample_rate) if new_sample_rate else 1.0
# #     return create_3d_scatter_figure(NODE_COORDS, NODE_INTENSITY, sample_rate=sample_rate)
# #
# #
# # @app.callback(
# #     Output('selected-voxel-store', 'data'),
# #     [
# #         Input('brain-3d-scatter', 'clickData'),
# #         Input('find-voxel-button', 'n_clicks')
# #     ],
# #     [
# #         State('x-input', 'value'),
# #         State('y-input', 'value'),
# #         State('z-input', 'value')
# #     ]
# # )
# # def store_voxel_index(clickData, find_n_clicks, x_val, y_val, z_val):
# #     """
# #     Set the selected voxel from either a 3D click or coordinate input.
# #     """
# #     from dash import callback_context as ctx
# #
# #     if not ctx.triggered:
# #         return None
# #
# #     triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
# #
# #     if triggered_id == 'brain-3d-scatter':
# #         if not clickData or not clickData['points']:
# #             return None
# #         voxel_index_str = clickData['points'][0].get('text')
# #         if voxel_index_str is None:
# #             return None
# #         return int(voxel_index_str)
# #
# #     elif triggered_id == 'find-voxel-button':
# #         coord_key = (int(x_val), int(y_val), int(z_val))
# #         voxel_idx = COORD_TO_INDEX.get(coord_key, None)
# #         return voxel_idx
# #
# #     return None
# #
# #
# # @app.callback(
# #     Output('time-series-plot', 'figure'),
# #     Input('selected-voxel-store', 'data')
# # )
# # def update_time_series_plot(voxel_index):
# #     """
# #     Update the time series plot when the selected voxel changes.
# #     """
# #     global BOLD_ARRAY, NODE_COORDS
# #     empty_fig = go.Figure().update_layout(
# #         title="Click a voxel or pick a coordinate to see BOLD time series",
# #         xaxis_title="Time",
# #         yaxis_title="BOLD"
# #     )
# #     if voxel_index is None or BOLD_ARRAY is None:
# #         return empty_fig
# #     if voxel_index < 0 or voxel_index >= BOLD_ARRAY.shape[0]:
# #         return empty_fig
# #
# #     time_series = BOLD_ARRAY[voxel_index]
# #     coordinate = NODE_COORDS[voxel_index]
# #     return create_time_series_figure(time_series, voxel_index, coordinate)
# #
# #
# # @app.callback(
# #     Output("download-dataframe-csv", "data"),
# #     Input("download-button", "n_clicks"),
# #     State("selected-voxel-store", "data"),
# #     prevent_initial_call=True
# # )
# # def download_csv(n_clicks, voxel_index):
# #     """
# #     Download a CSV of volume index vs. BOLD intensity for the selected voxel.
# #     """
# #     if voxel_index is None:
# #         raise dash.exceptions.PreventUpdate
# #
# #     global BOLD_ARRAY, NODE_COORDS
# #     if BOLD_ARRAY is None:
# #         raise dash.exceptions.PreventUpdate
# #     if voxel_index < 0 or voxel_index >= BOLD_ARRAY.shape[0]:
# #         raise dash.exceptions.PreventUpdate
# #
# #     ts = BOLD_ARRAY[voxel_index]
# #     coordinate = NODE_COORDS[voxel_index]
# #     df = pd.DataFrame({
# #         'volume_index': np.arange(len(ts)),
# #         'bold_intensity': ts
# #     })
# #     filename = f"voxel_{voxel_index}_{int(coordinate[0])}_{int(coordinate[1])}_{int(coordinate[2])}_bold.csv"
# #     return dcc.send_data_frame(df.to_csv, filename=filename, index=False)
# #
# #
# # @app.callback(
# #     Output("neighbour-table-div", "children"),
# #     Input("selected-voxel-store", "data")
# # )
# # def update_neighbour_table(voxel_index):
# #     """
# #     Update the neighbours table dynamically based on the selected voxel.
# #     For the chosen voxel, scan EDGE_INDEX to find all connected neighbours and
# #     list their (x,y,z) coordinates along with the corresponding edge weight.
# #     Duplicate neighbour entries are removed.
# #     """
# #     global NODE_COORDS, EDGE_INDEX, EDGE_WEIGHT
# #
# #     if voxel_index is None or EDGE_INDEX is None or EDGE_WEIGHT is None:
# #         return html.Div("Select a voxel to see its neighbours and edge weights.")
# #
# #     neighbours = {}
# #     for i in range(EDGE_INDEX.shape[1]):
# #         src = EDGE_INDEX[0, i]
# #         dst = EDGE_INDEX[1, i]
# #         if src == voxel_index or dst == voxel_index:
# #             neighbour_idx = dst if src == voxel_index else src
# #             neighbour_coord = tuple(map(int, NODE_COORDS[neighbour_idx]))
# #             if neighbour_coord not in neighbours:
# #                 neighbours[neighbour_coord] = EDGE_WEIGHT[i]
# #
# #     if not neighbours:
# #         return html.Div("No neighbouring voxels found for the selected voxel.")
# #
# #     header = [html.Tr([html.Th("Neighbour Voxel (x,y,z)"), html.Th("Edge Weight")])]
# #     rows = [html.Tr([html.Td(f"({n[0]}, {n[1]}, {n[2]})"), html.Td(f"{weight:.3f}")])
# #             for n, weight in neighbours.items()]
# #     table = html.Table(header + rows, style={
# #         'width': '100%',
# #         'border': '1px solid black',
# #         'borderCollapse': 'collapse',
# #         'marginTop': '10px'
# #     })
# #
# #     return table
# #
# #
# # def main():
# #     parser = argparse.ArgumentParser(
# #         description="Dash-based fMRI GNN Visualizer with coordinate picking, CSV download, and neighbour table.")
# #     parser.add_argument("--path", type=str, required=True,
# #                         help="Path to the saved PyG Data .pt file (shape: [num_voxels, 3 + T]).")
# #     parser.add_argument("--port", type=int, default=8050, help="Port for the Dash server.")
# #     parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for the Dash server.")
# #     parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode.")
# #     args = parser.parse_args()
# #
# #     # Load the PyG Data object
# #     print("Loading the PyG Data ...")
# #     data = torch.load(args.path, weights_only=False)
# #     if not isinstance(data, Data):
# #         raise ValueError("Loaded file is not a PyG Data object.")
# #
# #     node_features = data.x.cpu().numpy()  # shape: [N, 3 + T]
# #     if node_features.shape[1] < 4:
# #         raise ValueError("Expecting [x, y, z] + >=1 BOLD columns (total >=4 columns).")
# #
# #     global NODE_COORDS, NODE_INTENSITY, BOLD_ARRAY, COORD_TO_INDEX, EDGE_INDEX, EDGE_WEIGHT
# #     N = node_features.shape[0]
# #     coords = node_features[:, :3]
# #     bold_data = node_features[:, 3:]
# #
# #     # For coloring the 3D scatter, use the average BOLD if multiple timepoints
# #     if bold_data.shape[1] == 1:
# #         intensity = bold_data[:, 0]
# #     else:
# #         intensity = bold_data.mean(axis=1)
# #
# #     NODE_COORDS = coords
# #     NODE_INTENSITY = intensity
# #     BOLD_ARRAY = bold_data
# #
# #     # Build a dictionary for coordinate -> node index (assumes integer voxel indices)
# #     COORD_TO_INDEX = {}
# #     for idx in range(N):
# #         x, y, z = coords[idx].astype(int)
# #         COORD_TO_INDEX[(x, y, z)] = idx
# #
# #     # Load edge information: edge_index and edge_weight.
# #     EDGE_INDEX = data.edge_index.cpu().numpy()
# #     EDGE_WEIGHT = data.edge_weight.numpy()
# #
# #     print(f"Loaded graph with {N} voxels, each having {bold_data.shape[1]} timepoints.")
# #     print(f"Graph connectivity: {EDGE_INDEX.shape[1]} edges loaded.")
# #     print(f"Running Dash at http://{args.host}:{args.port}")
# #     app.run(host=args.host, port=args.port, debug=args.debug)
# #
# #
# # if __name__ == "__main__":
# #     main()
#
# # #!/usr/bin/env python3
# # import argparse
# # import numpy as np
# # import pandas as pd
# # import torch
# # from torch_geometric.data import Data
# #
# # import dash
# # from dash import dcc, html
# # from dash.dependencies import Input, Output, State
# # import plotly.graph_objs as go
# #
# # # ----------------------------------------------------------------------------
# # # Utility: parse FreeSurferColorLUT.txt to map ROI id -> name
# # # ----------------------------------------------------------------------------
# # def parse_lut(lut_path):
# #     lut = {}
# #     with open(lut_path, 'r') as f:
# #         for line in f:
# #             line = line.strip()
# #             if not line or line.startswith('#'):
# #                 continue
# #             parts = line.split()
# #             if len(parts) < 6:
# #                 continue
# #             try:
# #                 idx = int(parts[0])
# #                 # last 4 tokens are RGBA, name is everything between id and RGBA
# #                 name = ' '.join(parts[1:-4])
# #                 lut[idx] = name
# #             except ValueError:
# #                 continue
# #     return lut
# #
# # # ----------------------------------------------------------------------------
# # # Global placeholders for graph data
# # # ----------------------------------------------------------------------------
# # NODE_COORDS    = None   # [N,3]
# # NODE_INTENSITY = None   # [N]
# # NODE_ROIS      = None   # [N]
# # BOLD_ARRAY     = None   # [N,T]
# # COORD_TO_INDEX = None   # {(x,y,z): idx}
# # EDGE_INDEX     = None   # [2,E]
# # EDGE_WEIGHT    = None   # [E]
# # DEFAULT_SAMPLE_RATE = 0.1
# #
# # # ----------------------------------------------------------------------------
# # # Dash app setup
# # # ----------------------------------------------------------------------------
# # app = dash.Dash(__name__)
# #
# # # ----------------------------------------------------------------------------
# # # Figure creation
# # # ----------------------------------------------------------------------------
# # def create_3d_scatter_figure(coords, intensity, rois,
# #                              sample_rate=1.0, color_by='intensity', selected_rois=None):
# #     if selected_rois is not None:
# #         mask = np.isin(rois, selected_rois)
# #     else:
# #         mask = np.ones(rois.shape[0], dtype=bool)
# #     coords_f = coords[mask]
# #     inten_f  = intensity[mask]
# #     rois_f   = rois[mask]
# #     ids_f    = np.nonzero(mask)[0]
# #     Nf = coords_f.shape[0]
# #     if sample_rate < 1.0:
# #         k   = int(Nf * sample_rate)
# #         idx = np.random.choice(Nf, k, replace=False)
# #     else:
# #         idx = np.arange(Nf)
# #     coords_p = coords_f[idx]
# #     inten_p  = inten_f[idx]
# #     rois_p   = rois_f[idx]
# #     ids_p    = ids_f[idx]
# #     if color_by == 'roi':
# #         color_vals, colorscale, cbar_title = rois_p, 'Jet', 'ROI ID'
# #     else:
# #         color_vals, colorscale, cbar_title = inten_p, 'Viridis', 'Intensity'
# #     scatter = go.Scatter3d(
# #         x=coords_p[:,0], y=coords_p[:,1], z=coords_p[:,2],
# #         mode='markers',
# #         marker=dict(size=3, color=color_vals,
# #                     colorscale=colorscale, opacity=0.8,
# #                     showscale=True, colorbar=dict(title=cbar_title)),
# #         customdata=np.stack([ids_p, rois_p], axis=1),
# #         text=[str(i) for i in ids_p],
# #         hovertemplate=(
# #             "Idx: %{customdata[0]}<br>"
# #             "ROI ID: %{customdata[1]}<br>"
# #             "X: %{x} Y: %{y} Z: %{z}<extra></extra>"
# #         )
# #     )
# #     fig = go.Figure(data=[scatter])
# #     fig.update_layout(
# #         scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
# #         title='BOLD fMRI Graph (Voxel 3D)', clickmode='event+select'
# #     )
# #     return fig
# #
# #
# # def create_time_series_figure(ts, idx, coord):
# #     coord_str = f"({int(coord[0])},{int(coord[1])},{int(coord[2])})"
# #     fig = go.Figure()
# #     fig.add_trace(go.Scatter(x=np.arange(ts.shape[0]), y=ts, mode='lines',
# #                              name=f"Voxel {idx}"))
# #     fig.update_layout(title=f"BOLD Time Series: Voxel {idx} {coord_str}",
# #                       xaxis_title='Time idx', yaxis_title='Intensity')
# #     return fig
# #
# # # ----------------------------------------------------------------------------
# # # Main: load data, parse LUT, build layout, define callbacks, run server
# # # ----------------------------------------------------------------------------
# # def main():
# #     parser = argparse.ArgumentParser(description='fMRI GNN Dash Visualizer')
# #     parser.add_argument('--path', required=True, help='PyG Data .pt file path')
# #     parser.add_argument('--lut_path', required=True,
# #                         help='Path to FreeSurferColorLUT.txt')
# #     parser.add_argument('--port', type=int, default=8050)
# #     parser.add_argument('--host', type=str, default='127.0.0.1')
# #     parser.add_argument('--debug', action='store_true')
# #     args = parser.parse_args()
# #
# #     # parse LUT
# #     lut = parse_lut(args.lut_path)
# #
# #     # load graph
# #     data = torch.load(args.path, map_location='cpu')
# #     if not isinstance(data, Data):
# #         raise ValueError('Loaded file is not a PyG Data object')
# #     feats = data.x.numpy()
# #     N = feats.shape[0]
# #     coords = feats[:, :3]
# #     bold   = feats[:, 3:]
# #     intensity = bold.mean(axis=1) if bold.shape[1] > 1 else bold[:, 0]
# #
# #     global NODE_COORDS, NODE_INTENSITY, NODE_ROIS, BOLD_ARRAY, COORD_TO_INDEX, EDGE_INDEX, EDGE_WEIGHT
# #     NODE_COORDS    = coords
# #     NODE_INTENSITY = intensity
# #     NODE_ROIS      = data.roi.numpy()
# #     BOLD_ARRAY     = bold
# #     COORD_TO_INDEX = {tuple(coords[i].astype(int)): i for i in range(N)}
# #     EDGE_INDEX     = data.edge_index.numpy()
# #     EDGE_WEIGHT    = data.edge_weight.numpy()
# #
# #     # build ROI checklist options
# #     roi_ids = sorted(set(NODE_ROIS.tolist()))
# #     roi_options = [
# #         {'label': f"{rid}: {lut.get(rid, 'Unknown')}", 'value': str(rid)}
# #         for rid in roi_ids
# #     ]
# #
# #     # layout
# #     app.layout = html.Div([
# #         html.H1('BOLD fMRI Graph Visualization', style={'textAlign':'center'}),
# #         html.Div([
# #             html.Div([dcc.Graph(id='brain-3d-scatter')], style={'flex':'2','padding':'10px'}),
# #             html.Div([
# #                 html.Label('Color by:'),
# #                 dcc.Dropdown(
# #                     id='color-by-dropdown',
# #                     options=[
# #                         {'label': 'Intensity', 'value': 'intensity'},
# #                         {'label': 'ROI', 'value': 'roi'}
# #                     ], value='intensity', clearable=False,
# #                     style={'margin-bottom':'10px'}
# #                 ),
# #                 html.Label('Select/Deselect All ROIs:'),
# #                 dcc.Checklist(
# #                     id='select-all-checkbox',
# #                     options=[{'label':'All','value':'all'}],
# #                     value=['all'],
# #                     style={'margin-bottom':'10px'}
# #                 ),
# #                 html.Label('Filter ROIs:'),
# #                 dcc.Checklist(
# #                     id='roi-filter',
# #                     options=roi_options,
# #                     value=[str(r) for r in roi_ids],
# #                     inputStyle={'margin-right':'5px','margin-left':'10px'},
# #                     style={'maxHeight':'200px','overflowY':'auto','margin-bottom':'20px'}
# #                 ),
# #                 html.H3('Manual Coordinate:'),
# #                 html.Div([
# #                     'x: ', dcc.Input(id='x-input', type='number', value=0, style={'width':'20%'}),
# #                     ' y: ', dcc.Input(id='y-input', type='number', value=0, style={'width':'20%'}),
# #                     ' z: ', dcc.Input(id='z-input', type='number', value=0, style={'width':'20%'}),
# #                     html.Button('Find', id='find-voxel-button', style={'margin-left':'10px'})
# #                 ], style={'margin-bottom':'20px'}),
# #                 html.Label('Sample rate (0.01–1.0):'),
# #                 dcc.Input(
# #                     id='sample-rate-input', type='number', value=DEFAULT_SAMPLE_RATE,
# #                     min=0.01, max=1.0, step=0.01, style={'margin-left':'10px','margin-right':'10px'}
# #                 ),
# #                 html.Button('Update Plot', id='update-button', style={'margin-bottom':'20px'}),
# #                 html.Button('Download CSV', id='download-button'),
# #                 dcc.Download(id='download-dataframe-csv')
# #             ], style={'flex':'1','padding':'10px'})
# #         ], style={'display':'flex','width':'100%'}),
# #         html.Hr(),
# #         html.H3("Selected Voxel's Time Series", style={'textAlign':'center'}),
# #         dcc.Graph(id='time-series-plot'),
# #         dcc.Store(id='selected-voxel-store')
# #     ])
# #
# #     # callbacks
# #     @app.callback(
# #         Output('roi-filter','value'),
# #         Input('select-all-checkbox','value'),
# #         State('roi-filter','options')
# #     )
# #     def toggle_all(select_all, options):
# #         if 'all' in select_all:
# #             return [opt['value'] for opt in options]
# #         return []
# #
# #     @app.callback(
# #         Output('brain-3d-scatter','figure'),
# #         [Input('update-button','n_clicks'),
# #          Input('color-by-dropdown','value'),
# #          Input('roi-filter','value')],
# #         State('sample-rate-input','value')
# #     )
# #     def update_3d(nc, color_by, roi_vals, sample_rate):
# #         sel = [int(v) for v in (roi_vals or [])]
# #         return create_3d_scatter_figure(
# #             NODE_COORDS, NODE_INTENSITY, NODE_ROIS,
# #             sample_rate=float(sample_rate or 1.0),
# #             color_by=color_by, selected_rois=sel
# #         )
# #
# #     @app.callback(
# #         Output('selected-voxel-store','data'),
# #         [Input('brain-3d-scatter','clickData'), Input('find-voxel-button','n_clicks')],
# #         [State('x-input','value'), State('y-input','value'), State('z-input','value')]
# #     )
# #     def select_voxel(clickData, *_coords):
# #         from dash import callback_context as ctx
# #         if not ctx.triggered:
# #             return None
# #         trig = ctx.triggered[0]['prop_id'].split('.')[0]
# #         if trig == 'brain-3d-scatter' and clickData and clickData['points']:
# #             pt = clickData['points'][0]
# #             return int(pt.get('customdata',[None])[0] or pt.get('text'))
# #         if trig == 'find-voxel-button':
# #             x,y,z = _coords
# #             return COORD_TO_INDEX.get((int(x),int(y),int(z)))
# #         return None
# #
# #     @app.callback(
# #         Output('time-series-plot','figure'),
# #         Input('selected-voxel-store','data')
# #     )
# #     def show_ts(idx):
# #         if idx is None or BOLD_ARRAY is None or idx<0 or idx>=BOLD_ARRAY.shape[0]:
# #             return go.Figure().update_layout(title='Select a voxel')
# #         return create_time_series_figure(BOLD_ARRAY[idx], idx, NODE_COORDS[idx])
# #
# #     @app.callback(
# #         Output('download-dataframe-csv','data'),
# #         Input('download-button','n_clicks'),
# #         State('selected-voxel-store','data'),
# #         prevent_initial_call=True
# #     )
# #     def download_csv(nc, idx):
# #         if idx is None or BOLD_ARRAY is None:
# #             raise dash.exceptions.PreventUpdate
# #         ts, coord = BOLD_ARRAY[idx], NODE_COORDS[idx]
# #         df = pd.DataFrame({'volume': np.arange(ts.shape[0]), 'intensity': ts})
# #         fname = f"voxel_{idx}_{int(coord[0])}_{int(coord[1])}_{int(coord[2])}.csv"
# #         return dcc.send_data_frame(df.to_csv, filename=fname, index=False)
# #
# #     @app.callback(
# #         Output('neighbour-table-div','children'),
# #         Input('selected-voxel-store','data')
# #     )
# #     def show_neighbours(idx):
# #         if idx is None or EDGE_INDEX is None or EDGE_WEIGHT is None:
# #             return html.Div('Select voxel')
# #         nbrs = {}
# #         for e in range(EDGE_INDEX.shape[1]):
# #             s,d = EDGE_INDEX[0,e], EDGE_INDEX[1,e]
# #             if s==idx or d==idx:
# #                 nb = d if s==idx else s
# #                 c = tuple(map(int, NODE_COORDS[nb]))
# #                 if c not in nbrs:
# #                     nbrs[c] = (float(EDGE_WEIGHT[e]), int(NODE_ROIS[nb]))
# #         if not nbrs:
# #             return html.Div('No neighbours')
# #         header = [html.Tr([html.Th('Coord'), html.Th('Weight'), html.Th('ROI')])]
# #         rows = [html.Tr([html.Td(f"({c[0]},{c[1]},{c[2]})"), html.Td(f"{w:.3f}"), html.Td(str(r))])
# #                 for c,(w,r) in nbrs.items()]
# #         return html.Table(header + rows, style={'width':'100%','border':'1px solid black'})
# #
# #     print(f"Loaded {N} voxels, {bold.shape[1]} timepoints, {EDGE_INDEX.shape[1]} edges.")
# #     print(f"Serving at http://{args.host}:{args.port}")
# #     app.run(host=args.host, port=args.port, debug=args.debug)
# #
# # if __name__ == '__main__':
# #     main()
#
#
# #!/usr/bin/env python3
# import argparse
# import numpy as np
# import pandas as pd
# import torch
# from torch_geometric.data import Data
#
# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output, State
# import plotly.graph_objs as go
#
# # ----------------------------------------------------------------------------
# # Utility: parse FreeSurferColorLUT.txt to map ROI id -> name and color
# # ----------------------------------------------------------------------------
# def parse_lut(lut_path):
#     lut_names = {}
#     lut_colors = {}
#     with open(lut_path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith('#'):
#                 continue
#             parts = line.split()
#             if len(parts) < 6:
#                 continue
#             try:
#                 idx = int(parts[0])
#             except ValueError:
#                 continue
#             # last four tokens are RGBA
#             rgba = list(map(int, parts[-4:]))
#             # name is tokens[1:-4]
#             name = ' '.join(parts[1:-4])
#             lut_names[idx] = name
#             lut_colors[idx] = f"rgb({rgba[0]},{rgba[1]},{rgba[2]})"
#     return lut_names, lut_colors
#
# # ----------------------------------------------------------------------------
# # Global placeholders for graph data
# # ----------------------------------------------------------------------------
# NODE_COORDS    = None   # [N,3]
# NODE_INTENSITY = None   # [N]
# NODE_ROIS      = None   # [N]
# BOLD_ARRAY     = None   # [N,T]
# COORD_TO_INDEX = None   # {(x,y,z): idx}
# EDGE_INDEX     = None   # [2,E]
# EDGE_WEIGHT    = None   # [E]
# ROI_NAME_MAP   = {}     # {id: name}
# ROI_COLOR_MAP  = {}     # {id: 'rgb(r,g,b)'}
# DEFAULT_SAMPLE_RATE = 0.1
#
# # ----------------------------------------------------------------------------
# # Dash app setup
# # ----------------------------------------------------------------------------
# app = dash.Dash(__name__)
#
# # ----------------------------------------------------------------------------
# # Figure creation
# # ----------------------------------------------------------------------------
# def create_3d_scatter_figure(coords, intensity, rois,
#                              sample_rate=1.0, color_by='intensity', selected_rois=None):
#     # apply ROI filter
#     if selected_rois is not None:
#         mask = np.isin(rois, selected_rois)
#     else:
#         mask = np.ones(rois.shape[0], dtype=bool)
#     coords_f = coords[mask]
#     inten_f  = intensity[mask]
#     rois_f   = rois[mask]
#     ids_f    = np.nonzero(mask)[0]
#     Nf = coords_f.shape[0]
#     # downsample
#     if sample_rate < 1.0:
#         k   = int(Nf * sample_rate)
#         idx = np.random.choice(Nf, k, replace=False)
#     else:
#         idx = np.arange(Nf)
#     coords_p = coords_f[idx]
#     inten_p  = inten_f[idx]
#     rois_p   = rois_f[idx]
#     ids_p    = ids_f[idx]
#     if color_by == 'roi':
#         # use ROI-specific RGB colors
#         colors = [ROI_COLOR_MAP.get(int(r), 'rgb(128,128,128)') for r in rois_p]
#         marker=dict(size=3, color=colors, opacity=0.8)
#     else:
#         colors = inten_p
#         marker=dict(size=3, color=colors,
#                     colorscale='Viridis', opacity=0.8,
#                     showscale=True, colorbar=dict(title='Intensity'))
#     scatter = go.Scatter3d(
#         x=coords_p[:,0], y=coords_p[:,1], z=coords_p[:,2],
#         mode='markers', marker=marker,
#         customdata=np.stack([ids_p, rois_p], axis=1),
#         text=[str(i) for i in ids_p],
#         hovertemplate=(
#             "Idx: %{customdata[0]}<br>"
#             "ROI: %{customdata[1]}<br>"
#             "X: %{x} Y: %{y} Z: %{z}<extra></extra>"
#         )
#     )
#     fig = go.Figure(data=[scatter])
#     fig.update_layout(
#         scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
#         title='BOLD fMRI Graph (3D)', clickmode='event+select'
#     )
#     return fig
#
#
# def create_time_series_figure(ts, idx, coord):
#     coord_str = f"({int(coord[0])},{int(coord[1])},{int(coord[2])})"
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=np.arange(ts.shape[0]), y=ts, mode='lines',
#                              name=f"Voxel {idx}"))
#     fig.update_layout(title=f"BOLD Time Series: Voxel {idx} {coord_str}",
#                       xaxis_title='Time idx', yaxis_title='Intensity')
#     return fig
#
# # ----------------------------------------------------------------------------
# # Layout and Callbacks placeholder (built in main)
# # ----------------------------------------------------------------------------
#
# # ----------------------------------------------------------------------------
# # Main: load data, parse LUT, build layout, run server
# # ----------------------------------------------------------------------------
# def main():
#     parser = argparse.ArgumentParser(description='fMRI GNN Dash Visualizer')
#     parser.add_argument('--path',     required=True, help='PyG Data .pt file path')
#     parser.add_argument('--lut_path', required=True, help='Path to FreeSurferColorLUT.txt')
#     parser.add_argument('--port',     type=int, default=8050)
#     parser.add_argument('--host',     type=str, default='127.0.0.1')
#     parser.add_argument('--debug',    action='store_true')
#     args = parser.parse_args()
#     # parse LUT
#     names, colors = parse_lut(args.lut_path)
#     global ROI_NAME_MAP, ROI_COLOR_MAP
#     ROI_NAME_MAP  = names
#     ROI_COLOR_MAP = colors
#     # load graph
#     data = torch.load(args.path, map_location='cpu')
#     if not isinstance(data, Data): raise ValueError('Not PyG Data')
#     feats = data.x.numpy(); N = feats.shape[0]
#     coords = feats[:,:3]; bold = feats[:,3:]
#     inten  = bold.mean(axis=1) if bold.shape[1]>1 else bold[:,0]
#     global NODE_COORDS, NODE_INTENSITY, NODE_ROIS, BOLD_ARRAY, COORD_TO_INDEX, EDGE_INDEX, EDGE_WEIGHT
#     NODE_COORDS    = coords
#     NODE_INTENSITY = inten
#     NODE_ROIS      = data.roi.numpy()
#     BOLD_ARRAY     = bold
#     COORD_TO_INDEX = {tuple(coords[i].astype(int)):i for i in range(N)}
#     EDGE_INDEX     = data.edge_index.numpy()
#     EDGE_WEIGHT    = data.edge_weight.numpy()
#     # build ROI checklist options
#     roi_ids = sorted(set(NODE_ROIS.tolist()))
#     roi_options = [
#         {'label': f"{rid}: {ROI_NAME_MAP.get(rid,'Unknown')}", 'value': str(rid)}
#         for rid in roi_ids
#     ]
#     # define layout
#     app.layout = html.Div([
#         html.H1('BOLD fMRI Graph Visualization', style={'textAlign':'center'}),
#         html.Div([
#             html.Div([dcc.Graph(id='brain-3d-scatter')], style={'flex':'2','padding':'10px'}),
#             html.Div([
#                 html.Label('Color by:'),
#                 dcc.Dropdown(id='color-by-dropdown', options=[
#                     {'label':'Intensity','value':'intensity'},
#                     {'label':'ROI','value':'roi'}], value='intensity', clearable=False),
#                 html.Br(),
#                 dcc.Checklist(id='select-all-rois', options=[{'label':'Select All','value':'all'}], value=['all']),
#                 html.Label('Filter ROIs:'),
#                 dcc.Checklist(id='roi-filter', options=roi_options,
#                               value=[str(r) for r in roi_ids],
#                               inputStyle={'margin-right':'5px','margin-left':'10px'},
#                               style={'maxHeight':'200px','overflowY':'auto'}),
#                 html.Br(),
#                 html.H3('Manual Coord:'),
#                 'x: ',dcc.Input(id='x-input',type='number',value=0,style={'width':'20%'}),
#                 ' y: ',dcc.Input(id='y-input',type='number',value=0,style={'width':'20%'}),
#                 ' z: ',dcc.Input(id='z-input',type='number',value=0,style={'width':'20%'}),
#                 html.Button('Find',id='find-voxel-button'),
#                 html.Br(), html.Br(),
#                 html.Label('Sample rate:'),
#                 dcc.Input(id='sample-rate-input',type='number',value=DEFAULT_SAMPLE_RATE,
#                           min=0.01,max=1.0,step=0.01),
#                 html.Button('Update Plot',id='update-button'),
#                 html.Br(), html.Br(),
#                 html.Button('Download CSV',id='download-button'),
#                 dcc.Download(id='download-dataframe-csv')
#             ], style={'flex':'1','padding':'10px'})
#         ], style={'display':'flex','width':'100%'}),
#         html.Hr(),
#         html.H3("Selected Voxel's Time Series", style={'textAlign':'center'}),
#         dcc.Graph(id='time-series-plot'),
#         dcc.Store(id='selected-voxel-store')
#     ])
#     # callbacks
#     @app.callback(
#         Output('roi-filter','value'),
#         Input('select-all-rois','value'),
#         State('roi-filter','options')
#     )
#     def toggle_all(selected, options):
#         if 'all' in selected:
#             return [opt['value'] for opt in options]
#         return []
#     @app.callback(
#         Output('brain-3d-scatter','figure'),
#         [Input('update-button','n_clicks'),Input('color-by-dropdown','value'),Input('roi-filter','value')],
#         State('sample-rate-input','value')
#     )
#     def update_3d(nc, color_by, roi_vals, sample_rate):
#         if NODE_COORDS is None: return go.Figure()
#         sel = [int(v) for v in (roi_vals or [])]
#         return create_3d_scatter_figure(NODE_COORDS,NODE_INTENSITY,NODE_ROIS,
#             sample_rate=float(sample_rate or 1.0),color_by=color_by,selected_rois=sel)
#     @app.callback(
#         Output('selected-voxel-store','data'),
#         [Input('brain-3d-scatter','clickData'),Input('find-voxel-button','n_clicks')],
#         [State('x-input','value'),State('y-input','value'),State('z-input','value')]
#     )
#     def select_voxel(cd,*coords):
#         from dash import callback_context as ctx
#         if not ctx.triggered: return None
#         trig = ctx.triggered[0]['prop_id'].split('.')[0]
#         if trig=='brain-3d-scatter' and cd and cd['points']:
#             pt=cd['points'][0]
#             return int(pt.get('customdata',[None])[0] or pt.get('text'))
#         if trig=='find-voxel-button':
#             x,y,z=coords; return COORD_TO_INDEX.get((int(x),int(y),int(z)))
#         return None
#     @app.callback(Output('time-series-plot','figure'),Input('selected-voxel-store','data'))
#     def show_ts(i):
#         if i is None or BOLD_ARRAY is None: return go.Figure().update_layout(title='Select voxel')
#         if i<0 or i>=BOLD_ARRAY.shape[0]: return go.Figure().update_layout(title='Invalid voxel')
#         return create_time_series_figure(BOLD_ARRAY[i],i,NODE_COORDS[i])
#     @app.callback(Output('download-dataframe-csv','data'),Input('download-button','n_clicks'),State('selected-voxel-store','data'),prevent_initial_call=True)
#     def dl(n,i):
#         if i is None or BOLD_ARRAY is None: raise dash.exceptions.PreventUpdate
#         ts,coord=BOLD_ARRAY[i],NODE_COORDS[i]
#         df=pd.DataFrame({'volume':np.arange(ts.shape[0]),'intensity':ts})
#         fname=f"voxel_{i}_{int(coord[0])}_{int(coord[1])}_{int(coord[2])}.csv"
#         return dcc.send_data_frame(df.to_csv,fname,index=False)
#     @app.callback(Output('neighbour-table-div','children'),Input('selected-voxel-store','data'))
#     def show_nb(i):
#         if i is None or EDGE_INDEX is None or EDGE_WEIGHT is None: return html.Div('Select voxel')
#         nbrs={}
#         for e in range(EDGE_INDEX.shape[1]):
#             s,d=EDGE_INDEX[0,e],EDGE_INDEX[1,e]
#             if s==i or d==i:
#                 nb=d if s==i else s; c=tuple(map(int,NODE_COORDS[nb]))
#                 if c not in nbrs: nbrs[c]=(float(EDGE_WEIGHT[e]),int(NODE_ROIS[nb]))
#         if not nbrs: return html.Div('No neighbours')
#         hdr=[html.Tr([html.Th('Coord'),html.Th('Weight'),html.Th('ROI')])]
#         rows=[html.Tr([html.Td(f"({c[0]},{c[1]},{c[2]})"),html.Td(f"{w:.3f}"),html.Td(str(r))])
#                for c,(w,r) in nbrs.items()]
#         return html.Table(hdr+rows,style={'width':'100%','border':'1px solid black'})
#     # run
#     print(f"Loaded {N} voxels, {bold.shape[1]} timepoints, {EDGE_INDEX.shape[1]} edges.")
#     print(f"Serving at http://{args.host}:{args.port}")
#     app.run(host=args.host, port=args.port, debug=args.debug)
#
# if __name__=='__main__':
#     main()

#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

# ----------------------------------------------------------------------------
# Utility: parse FreeSurferColorLUT.txt to map ROI id -> name and color
# ----------------------------------------------------------------------------
def parse_lut(lut_path):
    lut_names = {}
    lut_colors = {}
    with open(lut_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            rgba = list(map(int, parts[-4:]))
            name = ' '.join(parts[1:-4])
            lut_names[idx] = name
            lut_colors[idx] = f'rgb({rgba[0]},{rgba[1]},{rgba[2]})'
    return lut_names, lut_colors

# ----------------------------------------------------------------------------
# Global placeholders for graph data
# ----------------------------------------------------------------------------
NODE_COORDS    = None   # [N,3]
NODE_INTENSITY = None   # [N]
NODE_ROIS      = None   # [N]
BOLD_ARRAY     = None   # [N,T]
COORD_TO_INDEX = None   # {(x,y,z): idx}
EDGE_INDEX     = None   # [2,E]
EDGE_WEIGHT    = None   # [E]
ROI_NAME_MAP   = {}     # {id: name}
ROI_COLOR_MAP  = {}     # {id: 'rgb(r,g,b)'}
DEFAULT_SAMPLE_RATE = 0.1

# ----------------------------------------------------------------------------
# Dash app setup
# ----------------------------------------------------------------------------
app = dash.Dash(__name__)

# ----------------------------------------------------------------------------
# Figure creation
# ----------------------------------------------------------------------------
def create_3d_scatter_figure(coords, intensity, rois,
                             sample_rate=1.0, color_by='intensity', selected_rois=None):
    if selected_rois is not None:
        mask = np.isin(rois, selected_rois)
    else:
        mask = np.ones(rois.shape[0], dtype=bool)

    coords_f = coords[mask]
    inten_f  = intensity[mask]
    rois_f   = rois[mask]
    ids_f    = np.nonzero(mask)[0]
    Nf = coords_f.shape[0]

    if sample_rate < 1.0:
        k = int(Nf * sample_rate)
        idxs = np.random.choice(Nf, k, replace=False)
    else:
        idxs = np.arange(Nf)

    coords_p = coords_f[idxs]
    inten_p  = inten_f[idxs]
    rois_p   = rois_f[idxs]
    ids_p    = ids_f[idxs]

    if color_by == 'roi':
        colors = [ROI_COLOR_MAP.get(int(r), 'rgb(128,128,128)') for r in rois_p]
        marker = dict(size=3, color=colors, opacity=0.8)
    else:
        colors = inten_p
        marker = dict(size=3, color=colors,
                      colorscale='Viridis', opacity=0.8,
                      showscale=True, colorbar=dict(title='Intensity'))

    scatter = go.Scatter3d(
        x=coords_p[:,0], y=coords_p[:,1], z=coords_p[:,2],
        mode='markers', marker=marker,
        customdata=np.stack([ids_p, rois_p], axis=1),
        text=[str(i) for i in ids_p],
        hovertemplate=(
            'Idx: %{customdata[0]}<br>'
            'ROI: %{customdata[1]}<br>'
            'X: %{x} Y: %{y} Z: %{z}<extra></extra>'
        )
    )
    fig = go.Figure(data=[scatter])
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        title='BOLD fMRI Graph (3D)', clickmode='event+select'
    )
    return fig

def create_time_series_figure(ts, idx, coord):
    coord_str = f'({int(coord[0])},{int(coord[1])},{int(coord[2])})'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(ts.shape[0]), y=ts, mode='lines',
                             name=f'Voxel {idx}'))
    fig.update_layout(title=f'BOLD Time Series: Voxel {idx} {coord_str}',
                      xaxis_title='Time idx', yaxis_title='Intensity')
    return fig

# ----------------------------------------------------------------------------
# Main: load data, parse LUT, build layout, define callbacks, run server
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='fMRI GNN Dash Visualizer')
    parser.add_argument('--path',     required=True, help='PyG Data .pt file path')
    parser.add_argument('--lut_path', required=True, help='Path to FreeSurferColorLUT.txt')
    parser.add_argument('--port',     type=int, default=8050)
    parser.add_argument('--host',     type=str, default='127.0.0.1')
    parser.add_argument('--debug',    action='store_true')
    args = parser.parse_args()

    # parse LUT
    names, colors = parse_lut(args.lut_path)
    global ROI_NAME_MAP, ROI_COLOR_MAP
    ROI_NAME_MAP  = names
    ROI_COLOR_MAP = colors

    # load graph
    data = torch.load(args.path, map_location='cpu')
    if not isinstance(data, Data):
        raise ValueError('Not a PyG Data object')
    feats = data.x.numpy(); N = feats.shape[0]
    coords = feats[:, :3]
    bold  = feats[:, 3:]
    inten  = bold.mean(axis=1) if bold.shape[1] > 1 else bold[:,0]

    global NODE_COORDS, NODE_INTENSITY, NODE_ROIS, BOLD_ARRAY, COORD_TO_INDEX, EDGE_INDEX, EDGE_WEIGHT
    NODE_COORDS    = coords
    NODE_INTENSITY = inten
    NODE_ROIS      = data.roi.numpy()
    BOLD_ARRAY     = bold
    COORD_TO_INDEX = {tuple(coords[i].astype(int)): i for i in range(N)}
    EDGE_INDEX     = data.edge_index.numpy()
    EDGE_WEIGHT    = data.edge_weight.numpy()

    # ROI filter options
    roi_ids = sorted(set(NODE_ROIS.tolist()))
    roi_options = [
        {'label': f"{rid}: {ROI_NAME_MAP.get(rid,'Unknown')}", 'value': str(rid)}
        for rid in roi_ids
    ]

    # layout
    app.layout = html.Div([
        html.H1('BOLD fMRI Graph Visualization', style={'textAlign':'center'}),
        html.Div([
            html.Div([dcc.Graph(id='brain-3d-scatter')], style={'flex':'2','padding':'10px'}),
            html.Div([
                html.Label('Color by:'),
                dcc.Dropdown(id='color-by-dropdown', options=[
                    {'label':'Intensity','value':'intensity'},
                    {'label':'ROI','value':'roi'}], value='intensity', clearable=False),
                html.Br(),
                dcc.Checklist(id='select-all-rois',
                              options=[{'label':'Select All','value':'all'}],
                              value=['all']),
                html.Label('Filter ROIs:'),
                dcc.Checklist(id='roi-filter',
                              options=roi_options,
                              value=[str(r) for r in roi_ids],
                              inputStyle={'margin-right':'5px','margin-left':'10px'},
                              style={'maxHeight':'200px','overflowY':'auto'}),
                html.Br(),

                html.H3('Neighbour Voxels and Edge Weights:'),
                html.Div(
                    id='neighbour-table-div',
                    style={
                        'margin-bottom': '20px',
                        'maxHeight': '240px',
                        'overflowY': 'auto',
                        'border': '1px solid #ccc',
                        'padding': '5px'
                    }
                ),

                html.H3('Manual Coord:'),
                'x:', dcc.Input(id='x-input', type='number', value=0, style={'width':'20%'}),
                ' y:', dcc.Input(id='y-input', type='number', value=0, style={'width':'20%'}),
                ' z:', dcc.Input(id='z-input', type='number', value=0, style={'width':'20%'}),
                html.Button('Find', id='find-voxel-button'),
                html.Br(), html.Br(),

                html.Label('Sample rate:'),
                dcc.Input(id='sample-rate-input',
                          type='number',
                          value=DEFAULT_SAMPLE_RATE,
                          min=0.01, max=1.0, step=0.01),
                html.Button('Update Plot', id='update-button'),
                html.Br(), html.Br(),

                html.Button('Download CSV', id='download-button'),
                dcc.Download(id='download-dataframe-csv')
            ], style={'flex':'1','padding':'10px'})
        ], style={'display':'flex','width':'100%'}),
        html.Hr(),
        html.H3("Selected Voxel's Time Series", style={'textAlign':'center'}),
        dcc.Graph(id='time-series-plot'),
        dcc.Store(id='selected-voxel-store')
    ])

    # callbacks
    @app.callback(
        Output('roi-filter','value'),
        Input('select-all-rois','value'),
        State('roi-filter','options')
    )
    def toggle_all(selected, options):
        return [opt['value'] for opt in options] if 'all' in selected else []

    @app.callback(
        Output('brain-3d-scatter','figure'),
        [Input('update-button','n_clicks'),
         Input('color-by-dropdown','value'),
         Input('roi-filter','value')],
        State('sample-rate-input','value')
    )
    def update_3d(nc, color_by, roi_vals, sample_rate):
        if NODE_COORDS is None:
            return go.Figure()
        sel = [int(v) for v in (roi_vals or [])]
        return create_3d_scatter_figure(
            NODE_COORDS, NODE_INTENSITY, NODE_ROIS,
            sample_rate=float(sample_rate or 1.0),
            color_by=color_by, selected_rois=sel
        )

    @app.callback(
        Output('selected-voxel-store','data'),
        [Input('brain-3d-scatter','clickData'),
         Input('find-voxel-button','n_clicks')],
        [State('x-input','value'),
         State('y-input','value'),
         State('z-input','value')]
    )
    def select_voxel(clickData, *_coords):
        from dash import callback_context as ctx
        if not ctx.triggered:
            return None
        trig = ctx.triggered[0]['prop_id'].split('.')[0]
        if trig == 'brain-3d-scatter' and clickData and clickData['points']:
            pt = clickData['points'][0]
            return int(pt.get('customdata',[None])[0] or pt.get('text'))
        if trig == 'find-voxel-button':
            x, y, z = _coords
            return COORD_TO_INDEX.get((int(x),int(y),int(z)))
        return None

    @app.callback(
        Output('time-series-plot','figure'),
        Input('selected-voxel-store','data')
    )
    def show_ts(idx):
        if idx is None or BOLD_ARRAY is None or not (0 <= idx < BOLD_ARRAY.shape[0]):
            return go.Figure().update_layout(title='Select a voxel')
        return create_time_series_figure(BOLD_ARRAY[idx], idx, NODE_COORDS[idx])

    @app.callback(
        Output('download-dataframe-csv','data'),
        Input('download-button','n_clicks'),
        State('selected-voxel-store','data'),
        prevent_initial_call=True
    )
    def dl(n_clicks, idx):
        if idx is None or BOLD_ARRAY is None:
            raise dash.exceptions.PreventUpdate
        ts = BOLD_ARRAY[idx]
        coord = NODE_COORDS[idx]
        df = pd.DataFrame({'volume': np.arange(ts.shape[0]), 'intensity': ts})
        fname = f"voxel_{idx}_{int(coord[0])}_{int(coord[1])}_{int(coord[2])}.csv"
        return dcc.send_data_frame(df.to_csv, filename=fname, index=False)

    @app.callback(
        Output('neighbour-table-div','children'),
        Input('selected-voxel-store','data')
    )
    def show_neighbours(idx):
        if idx is None or EDGE_INDEX is None or EDGE_WEIGHT is None:
            return html.Div('Select voxel')
        neighbours = {}
        for e in range(EDGE_INDEX.shape[1]):
            s, d = EDGE_INDEX[0,e], EDGE_INDEX[1,e]
            if s == idx or d == idx:
                nb = d if s == idx else s
                coord = tuple(map(int, NODE_COORDS[nb]))
                if coord not in neighbours:
                    neighbours[coord] = (float(EDGE_WEIGHT[e]), int(NODE_ROIS[nb]))
        if not neighbours:
            return html.Div('No neighbours')

        header = [html.Tr([html.Th('Coord'), html.Th('Weight'), html.Th('ROI')])]
        rows = [
            html.Tr([
                html.Td(f"({c[0]},{c[1]},{c[2]})"),
                html.Td(f"{w:.3f}"),
                html.Td(str(r))
            ])
            for c, (w, r) in neighbours.items()
        ]
        table = html.Table(header + rows, style={'width':'100%', 'borderCollapse':'collapse'})

        return html.Div(
            table,
            style={
                'maxHeight': '240px',
                'overflowY': 'auto',
                'border': '1px solid #ccc',
                'padding': '5px'
            }
        )

    # run the server
    print(f"Loaded {N} voxels, {bold.shape[1]} timepoints, {EDGE_INDEX.shape[1]} edges.")
    print(f"Serving at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__=='__main__':
    main()

