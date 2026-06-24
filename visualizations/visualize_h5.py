import os
import h5py
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Disable HDF5 file locking to avoid BlockingIOError
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'


def parse_range_expression(expr: str, available_keys: list) -> list:
    """
    Parse a range expression like '20-26', '1,3,5-10', or '0,2-4,7'
    against a sorted list of available trajectory keys.
    Keys are matched by their numeric suffix or positional index.

    Returns a list of matching trajectory keys.
    """
    sorted_keys = sorted(available_keys)

    # Build an index → key map (by position) and a label → key map (by numeric suffix)
    position_map = {i: k for i, k in enumerate(sorted_keys)}

    # Try to extract trailing integer from key names, e.g. "traj_020" → 20
    import re
    label_map = {}
    for k in sorted_keys:
        m = re.search(r'(\d+)$', k)
        if m:
            label_map[int(m.group(1))] = k

    selected = set()

    for part in expr.split(','):
        part = part.strip()
        if not part:
            continue

        range_match = re.fullmatch(r'(\d+)\s*-\s*(\d+)', part)
        single_match = re.fullmatch(r'(\d+)', part)

        if range_match:
            lo, hi = int(range_match.group(1)), int(range_match.group(2))
            # Prefer label_map (numeric key suffix), fall back to position_map
            for n in range(lo, hi + 1):
                if n in label_map:
                    selected.add(label_map[n])
                elif n in position_map:
                    selected.add(position_map[n])
        elif single_match:
            n = int(single_match.group(1))
            if n in label_map:
                selected.add(label_map[n])
            elif n in position_map:
                selected.add(position_map[n])

    # Preserve original sort order
    return [k for k in sorted_keys if k in selected]


class HDF5Visualizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.trajectories = {}
        self.load_data()

    def load_data(self):
        print(f"Loading data from {self.file_path}...")
        try:
            with h5py.File(self.file_path, 'r', swmr=True) as f:
                for key in f.keys():
                    grp = f[key]
                    if isinstance(grp, h5py.Group):
                        traj_data = {
                            'states': np.array(grp['states']),
                            'actions': np.array(grp['actions']),
                            'attrs': {k: v for k, v in grp.attrs.items()}
                        }
                        for k, v in traj_data['attrs'].items():
                            if isinstance(v, bytes):
                                try:
                                    traj_data['attrs'][k] = v.decode('ascii')
                                except:
                                    pass
                        self.trajectories[key] = traj_data
            print(f"Loaded {len(self.trajectories)} trajectories.")
        except Exception as e:
            print(f"Error loading HDF5 file: {e}")

    def create_app(self):
        app = dash.Dash(__name__)

        traj_options = [{'label': k, 'value': k} for k in sorted(self.trajectories.keys())]
        sorted_keys = sorted(self.trajectories.keys())

        app.layout = html.Div([
            dcc.Store(id='last-selected', data=None),
            html.H1("HDF5 Expert Trajectory Visualizer", style={'textAlign': 'center'}),

            html.Div([
                # ── Left column: dropdown + quick-select ──────────────────────
                html.Div([
                    html.Label("Select Trajectory:"),
                    dcc.Dropdown(
                        id='traj-dropdown',
                        options=traj_options,
                        value=[traj_options[0]['value']] if traj_options else None,
                        multi=True,
                        placeholder='Select one or more trajectories to overlay'
                    ),

                    # Quick range selector
                    html.Div([
                        html.Label(
                            "Quick select (e.g. 20-26, 0,2,5-8):",
                            style={'marginTop': '12px', 'display': 'block', 'fontWeight': '500'}
                        ),
                        html.Div([
                            dcc.Input(
                                id='range-input',
                                type='text',
                                placeholder='e.g. 20-26 or 0,2,5-8',
                                debounce=False,
                                style={
                                    'flex': '1',
                                    'padding': '6px 10px',
                                    'fontSize': '14px',
                                    'border': '1px solid #ccc',
                                    'borderRadius': '4px 0 0 4px',
                                    'outline': 'none',
                                }
                            ),
                            html.Button(
                                'Apply',
                                id='range-apply-btn',
                                n_clicks=0,
                                style={
                                    'padding': '6px 14px',
                                    'fontSize': '14px',
                                    'border': '1px solid #4a90d9',
                                    'borderLeft': 'none',
                                    'borderRadius': '0 4px 4px 0',
                                    'backgroundColor': '#4a90d9',
                                    'color': 'white',
                                    'cursor': 'pointer',
                                }
                            ),
                            html.Button(
                                'Clear',
                                id='range-clear-btn',
                                n_clicks=0,
                                style={
                                    'padding': '6px 14px',
                                    'fontSize': '14px',
                                    'marginLeft': '6px',
                                    'border': '1px solid #aaa',
                                    'borderRadius': '4px',
                                    'backgroundColor': '#f5f5f5',
                                    'cursor': 'pointer',
                                }
                            ),
                        ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '6px'}),

                        # Feedback line
                        html.Div(
                            id='range-feedback',
                            style={'marginTop': '4px', 'fontSize': '12px', 'color': '#666', 'minHeight': '18px'}
                        ),
                    ]),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                # ── Right column: metadata ────────────────────────────────────
                html.Div([
                    html.Label("Metadata:"),
                    html.Pre(id='metadata-display', style={
                        'border': '1px solid #ccc',
                        'padding': '20px',
                        'height': '150px',
                        'overflowY': 'scroll',
                        'position': 'relative',
                        'zIndex': 10,
                        'backgroundColor': 'white',
                        'marginBottom': '20px'
                    })
                ], style={'width': '48%', 'display': 'inline-block', 'float': 'right', 'verticalAlign': 'top'}),
            ], style={'padding': '20px'}),

            html.Div([
                dcc.Graph(id='state-plot'),
                dcc.Graph(id='action-plot')
            ], style={'clear': 'both'}),

            html.Div(id='dummy-output', style={'display': 'none'})
        ])

        # ── Callback: Apply / Clear range → update dropdown ──────────────────
        @app.callback(
            Output('traj-dropdown', 'value'),
            Output('range-feedback', 'children'),
            Output('range-feedback', 'style'),
            Input('range-apply-btn', 'n_clicks'),
            Input('range-clear-btn', 'n_clicks'),
            State('range-input', 'value'),
            State('traj-dropdown', 'value'),
            prevent_initial_call=True
        )
        def apply_range(apply_clicks, clear_clicks, range_expr, current_value):
            ctx = dash.callback_context
            if not ctx.triggered:
                return current_value, '', {'fontSize': '12px', 'color': '#666', 'minHeight': '18px'}

            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if triggered_id == 'range-clear-btn':
                return [], 'Selection cleared.', {'fontSize': '12px', 'color': '#888', 'minHeight': '18px'}

            # Apply button
            if not range_expr or not range_expr.strip():
                return current_value, '⚠ Enter a range expression first.', \
                       {'fontSize': '12px', 'color': '#c0392b', 'minHeight': '18px'}

            matched = parse_range_expression(range_expr, list(self.trajectories.keys()))

            if not matched:
                return current_value, f'⚠ No trajectories matched "{range_expr}".', \
                       {'fontSize': '12px', 'color': '#c0392b', 'minHeight': '18px'}

            msg = f'✓ {len(matched)} trajectorie(s) selected: {matched[0]}…{matched[-1]}'
            return matched, msg, {'fontSize': '12px', 'color': '#27ae60', 'minHeight': '18px'}

        # ── Callback: render plots ────────────────────────────────────────────
        @app.callback(
            [Output('state-plot', 'figure'),
             Output('action-plot', 'figure'),
             Output('metadata-display', 'children')],
            [Input('traj-dropdown', 'value')]
        )
        def update_plots(selected_traj):
            if not selected_traj:
                return go.Figure(), go.Figure(), "No data"

            selected = selected_traj if isinstance(selected_traj, (list, tuple)) else [selected_traj]
            valid = [s for s in selected if s in self.trajectories]
            if not valid:
                return go.Figure(), go.Figure(), "No data"

            fig_states = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Position 1", "Position 2", "Velocity 1", "Velocity 2")
            )
            for s in valid:
                data = self.trajectories[s]
                states = data['states']
                t = np.arange(states.shape[0])
                fig_states.add_trace(go.Scatter(x=t, y=states[:, 0], name=f"{s} - q1", opacity=0.8), row=1, col=1)
                fig_states.add_trace(go.Scatter(x=t, y=states[:, 1], name=f"{s} - q2", opacity=0.8), row=1, col=2)
                fig_states.add_trace(go.Scatter(x=t, y=states[:, 2], name=f"{s} - v1", opacity=0.8), row=2, col=1)
                fig_states.add_trace(go.Scatter(x=t, y=states[:, 3], name=f"{s} - v2", opacity=0.8), row=2, col=2)

            title_sel = ", ".join(valid)
            fig_states.update_layout(height=600, title_text=f"States for {title_sel}")

            fig_actions = go.Figure()
            for s in valid:
                data = self.trajectories[s]
                actions = data['actions']
                ta = np.arange(actions.shape[0])
                for i in range(actions.shape[1]):
                    fig_actions.add_trace(go.Scatter(x=ta, y=actions[:, i], name=f"{s} - u{i+1}", opacity=0.8))

            fig_actions.update_layout(
                height=400,
                title_text=f"Actions for {title_sel}",
                xaxis_title="Step",
                yaxis_title="Torque"
            )

            metadata_blocks = []
            for s in valid:
                attrs = self.trajectories[s]['attrs']
                r_weight = attrs.get('R_weight', 'N/A')
                q_weights = attrs.get('Q_weights', 'N/A')
                peak_torque = attrs.get('peak_torque', 'N/A')
                status = "Y" if "successfully" in str(attrs.get('status', '')).lower() else "N"
                meta_row = (
                    f"{s:<8} R_weight: {str(r_weight):<7} "
                    f"Q_weights: {str(q_weights):<25} "
                    f"peak_torque: {str(peak_torque):<6} status: {status}"
                )
                metadata_blocks.append(meta_row)

            return fig_states, fig_actions, "\n".join(metadata_blocks).strip()

        return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize HDF5 trajectories")
    parser.add_argument("--file", type=str, default="double_pendulum/results/expert_trajectories.h5",
                        help="Path to HDF5 file")
    parser.add_argument("--port", type=int, default=8050, help="Port for Dash server")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}")
    else:
        visualizer = HDF5Visualizer(args.file)
        if not visualizer.trajectories:
            print("No trajectories found in file.")
        else:
            app = visualizer.create_app()
            print(f"Starting server on port {args.port}...")
            app.run(debug=True, port=args.port, host='0.0.0.0')