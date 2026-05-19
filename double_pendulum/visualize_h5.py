import os
import h5py
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Disable HDF5 file locking to avoid BlockingIOError
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

class HDF5Visualizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.trajectories = {}
        self.load_data()

    def load_data(self):
        print(f"Loading data from {self.file_path}...")
        try:
            # Try with swmr=True and locking disabled
            with h5py.File(self.file_path, 'r', swmr=True) as f:
                for key in f.keys():
                    grp = f[key]
                    if isinstance(grp, h5py.Group):
                        traj_data = {
                            'states': np.array(grp['states']),
                            'actions': np.array(grp['actions']),
                            'attrs': {k: v for k, v in grp.attrs.items()}
                        }
                        # Decode bytes attributes
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
                html.Div([
                    html.Label("Select Trajectory:"),
                    dcc.Dropdown(
                        id='traj-dropdown',
                        options=traj_options,
                        value=[traj_options[0]['value']] if traj_options else None,
                        multi=True,
                        placeholder='Select one or more trajectories to overlay'
                    ),
                ], style={'width': '48%', 'display': 'inline-block'}),
                
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
                ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
            ], style={'padding': '20px'}),

            # ensure graphs are rendered below the header/metadata area to avoid overlap
            html.Div([
                dcc.Graph(id='state-plot'),
                dcc.Graph(id='action-plot')
            ], style={'clear': 'both'}),
            
            html.Div(id='dummy-output', style={'display': 'none'})
        ])

        @app.callback(
            [Output('state-plot', 'figure'),
             Output('action-plot', 'figure'),
             Output('metadata-display', 'children')],
            [Input('traj-dropdown', 'value')]
        )
        def update_plots(selected_traj):
            if not selected_traj:
                return go.Figure(), go.Figure(), "No data"

            # allow single selection or multiple
            selected = selected_traj if isinstance(selected_traj, (list, tuple)) else [selected_traj]
            valid = [s for s in selected if s in self.trajectories]
            if not valid:
                return go.Figure(), go.Figure(), "No data"

            # State Plot (overlay multiple trajectories)
            fig_states = make_subplots(rows=2, cols=2, subplot_titles=("Position 1", "Position 2", "Velocity 1", "Velocity 2"))
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

            # Action Plot
            # Action Plot (overlay)
            fig_actions = go.Figure()
            for s in valid:
                data = self.trajectories[s]
                actions = data['actions']
                ta = np.arange(actions.shape[0])
                for i in range(actions.shape[1]):
                    fig_actions.add_trace(go.Scatter(x=ta, y=actions[:, i], name=f"{s} - u{i+1}", opacity=0.8))

            fig_actions.update_layout(height=400, title_text=f"Actions for {title_sel}", xaxis_title="Step", yaxis_title="Torque")

            # Metadata: show qr config and attributes for each selected trajectory
            metadata_blocks = []
            for s in valid:
                attrs = self.trajectories[s]['attrs']
                r_weight = attrs.get('R_weight', 'N/A')
                q_weights = attrs.get('Q_weights', 'N/A')
                peak_torque = attrs.get('peak_torque', 'N/A')
                status = "Y" if "successfully" in str(attrs.get('status', '')).lower() else "N"
                
                # Formatted string with padding for alignment
                meta_row = f"{s:<8} R_weight: {str(r_weight):<7} Q_weights: {str(q_weights):<25} peak_torque: {str(peak_torque):<6} status: {status}"
                metadata_blocks.append(meta_row)
            metadata_str = "\n".join(metadata_blocks).strip()

            return fig_states, fig_actions, metadata_str

        return app

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize HDF5 trajectories")
    parser.add_argument("--file", type=str, default="double_pendulum/results/expert_trajectories.h5", help="Path to HDF5 file")
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
