"""
Train network visualization module for generating time-space diagrams.
"""

import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Tuple

class TrainNetworkVisualizer:
    """Visualization tools for train network simulations."""
    
    def __init__(
        self,
        stations: List[str],
        time_range: Optional[Tuple[float, float]] = None,
        height: int = 800,
        width: int = 1200
    ):
        """
        Initialize the visualizer.
        
        Args:
            stations: List of station names in order
            time_range: Optional tuple of (start_time, end_time) for x-axis
            height: Plot height in pixels
            width: Plot width in pixels
        """
        self.stations = stations
        self.station_positions = {station: idx for idx, station in enumerate(stations)}
        self.time_range = time_range or (0, 1)
        self.height = height
        self.width = width
        self.colors = {
            'planned': 'rgba(0, 0, 255, 0.7)',
            'actual': 'rgba(255, 0, 0, 0.7)',
            'conflict': 'rgba(255, 165, 0, 0.7)'
        }

    def create_time_space_diagram(
        self,
        planned_schedule: Dict[str, List[Tuple]],
        actual_schedule: Optional[Dict[str, List[Tuple]]] = None,
        conflicts: Optional[List[Tuple[str, float]]] = None,
        title: str = "Train Network Time-Space Diagram"
    ) -> go.Figure:
        """
        Create an interactive time-space diagram.
        
        Args:
            planned_schedule: Dictionary of planned train movements
            actual_schedule: Optional dictionary of actual train movements
            conflicts: Optional list of (station, time) tuples marking conflicts
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add planned schedule traces
        self._add_schedule_traces(
            fig, 
            planned_schedule, 
            'Planned', 
            self.colors['planned'], 
            'solid'
        )
        
        # Add actual schedule traces if provided
        if actual_schedule:
            self._add_schedule_traces(
                fig, 
                actual_schedule, 
                'Actual', 
                self.colors['actual'], 
                'dash'
            )
            
        # Add conflict markers if provided
        if conflicts:
            self._add_conflict_markers(fig, conflicts)
            
        # Update layout
        fig.update_layout(
            title=title,
            xaxis=dict(
                title='Time',
                range=self.time_range,
                gridcolor='rgba(128, 128, 128, 0.2)',
                showgrid=True,
                dtick=0.01
            ),
            yaxis=dict(
                title='Stations',
                ticktext=self.stations,
                tickvals=list(range(len(self.stations))),
                gridcolor='rgba(128, 128, 128, 0.2)',
                showgrid=True
            ),
            height=self.height,
            width=self.width,
            plot_bgcolor='white',
            showlegend=True,
            hovermode='closest'
        )
        
        return fig

    def _add_schedule_traces(
        self,
        fig: go.Figure,
        schedule: Dict[str, List[Tuple]],
        name: str,
        color: str,
        line_style: str
    ):
        """Add schedule traces to the figure."""
        # Group movements by train
        train_paths = {}
        
        for station, movements in schedule.items():
            station_pos = self.station_positions[station]
            for train_id, arrival, departure, direction in movements:
                if train_id not in train_paths:
                    train_paths[train_id] = []
                train_paths[train_id].extend([
                    (arrival, station_pos),
                    (departure, station_pos)
                ])
        
        # Create traces for each train
        for train_id, path in train_paths.items():
            # Sort by time
            path.sort()
            times, positions = zip(*path)
            
            fig.add_trace(go.Scatter(
                x=times,
                y=positions,
                mode='lines',
                name=f'{name} Train {train_id}',
                line=dict(
                    color=color,
                    dash=line_style,
                    width=2
                ),
                hovertemplate=(
                    f'Train {train_id}<br>'
                    'Time: %{x:.3f}<br>'
                    'Station: %{text}<br>'
                ),
                text=[self.stations[pos] for pos in positions]
            ))

    def _add_conflict_markers(
        self,
        fig: go.Figure,
        conflicts: List[Tuple[str, float]]
    ):
        """Add conflict markers to the figure."""
        for station, time in conflicts:
            station_pos = self.station_positions[station]
            
            fig.add_trace(go.Scatter(
                x=[time],
                y=[station_pos],
                mode='markers',
                name='Conflict',
                marker=dict(
                    symbol='x',
                    size=12,
                    color=self.colors['conflict'],
                    line=dict(width=2)
                ),
                hovertemplate=(
                    'Conflict<br>'
                    'Time: %{x:.3f}<br>'
                    'Station: %{text}<br>'
                ),
                text=[station]
            ))

# Example usage
if __name__ == "__main__":
    # Example data
    stations = ['Station1', 'Station2', 'Station3', 'Station4']
    
    planned_schedule = {
        'Station1': [
            (1, 0.1, 0.12, 1),
            (2, 0.2, 0.22, 1)
        ],
        'Station2': [
            (1, 0.15, 0.17, 1),
            (2, 0.25, 0.27, 1)
        ],
        'Station3': [
            (1, 0.2, 0.22, 1),
            (2, 0.3, 0.32, 1)
        ],
        'Station4': [
            (1, 0.25, 0.27, 1),
            (2, 0.35, 0.37, 1)
        ]
    }
    
    actual_schedule = {
        'Station1': [
            (1, 0.11, 0.13, 1),
            (2, 0.21, 0.23, 1)
        ],
        'Station2': [
            (1, 0.16, 0.18, 1),
            (2, 0.26, 0.28, 1)
        ],
        'Station3': [
            (1, 0.21, 0.23, 1),
            (2, 0.31, 0.33, 1)
        ],
        'Station4': [
            (1, 0.26, 0.28, 1),
            (2, 0.36, 0.38, 1)
        ]
    }
    
    conflicts = [
        ('Station2', 0.27),
        ('Station3', 0.32)
    ]
    
    # Create visualization
    visualizer = TrainNetworkVisualizer(
        stations=stations,
        time_range=(0, 0.5)
    )
    
    fig = visualizer.create_time_space_diagram(
        planned_schedule=planned_schedule,
        actual_schedule=actual_schedule,
        conflicts=conflicts
    )
    
    fig.show()