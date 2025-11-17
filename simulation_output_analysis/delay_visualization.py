""" created by Paula von der Heide, TUBS, 2025

If you use this code in your research or publications, please cite: 
"A SUMO-based study pf Urban Rail Operations on Frankfurts Corridor A", Paula von der Heide und Prof. Dr.-Ing. Lars Schnieder, 5.th International Railway Symposium Aachen, 2025
not for commercial use """

import xml.etree.ElementTree as ET
import os
import plotly.graph_objects as go # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def plot_multi_vehicles(combined_map, stop_names_map=None):
    """
    Plot scheduled vs actual departure times for all vehicles.

    Args:
      combined_map   : dict { vehicle_id → list of entries }
                       each entry is {'busStop','scheduled','actual'}
      stop_names_map : dict { stopID → stopName }, or None to leave IDs
    """
    fig = go.Figure()

    for vid, entries in combined_map.items():
        if not entries:
            continue

        # extract parallel lists
        stops     = [e['busStop']   for e in entries]
        scheduled = [e['scheduled'] for e in entries]
        actual    = [(e['actual'][0] if e['actual'] else None)
                     for e in entries]

        # translate IDs to names if provided
        if stop_names_map:
            stops = [ stop_names_map.get(s, s) for s in stops ]

        # scheduled trace
        fig.add_trace(go.Scatter(
            x=stops,
            y=scheduled,
            mode='lines+markers',
            name=f'{vid} Scheduled'
        ))
        # actual trace (gaps shown)
        fig.add_trace(go.Scatter(
            x=stops,
            y=actual,
            mode='lines+markers',
            name=f'{vid} Actual'
        ))

    # time → y (inverted), stops → x
    fig.update_yaxes(autorange='reversed', title_text='Time (s)')
    fig.update_xaxes(title_text='Bus stop', tickangle=45)

    fig.update_layout(
        title='Scheduled vs Actual Departure Times',
        margin=dict(l=40, r=20, t=50, b=100),
        legend_title='Vehicle / Type'
    )

    return fig

def plot_delay_cumulative_recovered_and_added_interactive(
    combined_map,
    vehicle_ids=None,
    stop_names_map=None
):
    """
    Interactive plot of per-stop delay, cumulative recovered delay, and cumulative added delay 
    for one or more vehicles.

    Args:
      combined_map   : dict { vehicle_id → list of entries }
                       each entry has {'busStop','scheduled','actual':[...] }
      vehicle_ids    : list of vehicle IDs to plot; if None, plots all
      stop_names_map : dict { stopID → stopName }, or None to use raw IDs

    Returns:
      A Plotly Figure object (call .show() to display).
    """
    # Default to all vehicles if none specified
    if vehicle_ids is None:
        vehicle_ids = list(combined_map.keys())

    fig = go.Figure()

    for vid in vehicle_ids:
        entries = combined_map.get(vid, [])
        if not entries:
            continue

        # Extract stops and compute per-stop delays
        stops = [e['busStop'] for e in entries]
        delays = [(e['actual'][0] - e['scheduled']) if e['actual'] else None
                  for e in entries]

        # Compute per-stop recovered and added
        per_stop_recovered = [0] * len(delays)
        per_stop_added = [0] * len(delays)
        for i in range(1, len(delays)):
            prev = delays[i-1]
            curr = delays[i]
            if prev is not None and curr is not None:
                if prev > curr:
                    per_stop_recovered[i] = prev - curr
                    per_stop_added[i] = 0
                elif curr > prev:
                    per_stop_added[i] = curr - prev
                    per_stop_recovered[i] = 0
                else:
                    per_stop_added[i] = 0
                    per_stop_recovered[i] = 0

        # Build cumulative recovered and cumulative added arrays
        cumulative_recovered = []
        cumulative_added = []
        running_rec = 0
        running_add = 0
        for rec, add in zip(per_stop_recovered, per_stop_added):
            running_rec += rec
            running_add += add
            cumulative_recovered.append(running_rec)
            cumulative_added.append(running_add)

        # Map IDs to names if provided
        if stop_names_map:
            stops = [stop_names_map.get(s, s) for s in stops]

        # Add per-stop delay trace
        fig.add_trace(go.Scatter(
            x=stops,
            y=delays,
            mode='lines+markers',
            name=f'{vid} Delay'
        ))
        # Add cumulative recovered trace
        fig.add_trace(go.Scatter(
            x=stops,
            y=cumulative_recovered,
            mode='lines+markers',
            name=f'{vid} Cumulative Recovered'
        ))
        # Add cumulative added trace
        fig.add_trace(go.Scatter(
            x=stops,
            y=cumulative_added,
            mode='lines+markers',
            name=f'{vid} Cumulative Added'
        ))

    # Final layout tweaks
    fig.update_xaxes(title_text='Bus stop', tickangle=45)
    fig.update_yaxes(title_text='Time (s)')
    fig.update_layout(
        title='Per-Stop Delay, Cumulative Recovered, and Cumulative Added Delay',
        margin=dict(l=40, r=20, t=50, b=100),
        legend_title='Vehicle / Metric'
    )
    return fig

def plot_top_n_delayed_stations(stats_df, stop_names_map, top_n=10):
    """
    Interactive horizontal bar chart of the top N stops by average delay,
    labeling each with "stop_id – stop_name".

    Args:
      stats_df       : pandas DataFrame with columns ['stop_id','mean'].
      stop_names_map : dict mapping stop_id to stop_name.
      top_n          : int, number of stops to display.
    """
    # 1) Sort by mean delay descending
    df_sorted = stats_df.sort_values('mean', ascending=False)

    # 2) Take top N rows
    df_plot = df_sorted.head(top_n)

    # 3) Build combined labels "id – name"
    stop_ids = df_plot['stop_id'].astype(str)
    labels = [f"{sid} – {stop_names_map.get(sid, sid)}" for sid in stop_ids]

    # 4) Values are the mean delays
    values = df_plot['mean']

    # 5) Plot interactive horizontal bar chart
    fig = go.Figure(go.Bar(
        x=values[::-1],
        y=labels[::-1],
        orientation='h',
        marker=dict(line=dict(width=1, color='black')),
        name='Average Delay'
    ))

    fig.update_layout(
        title=f'Top {len(df_plot)} Stops by Average Delay',
        xaxis_title='Average Delay (s)',
        yaxis_title='Stop ID – Stop Name',
        margin=dict(l=150, r=20, t=50, b=50),
        template='plotly_white'
    )
    return fig

def plot_station_delay_hist(combined_map, bins=50):
    """
    Interactive histogram of all delays as percentages of total, per bin.

    Args:
      combined_map : dict { vehicle_id → list of entries }
                     each entry has 'scheduled' and 'actual': [...]
      bins         : int, number of histogram bins
    """
    # Flatten all delay values
    delays = []
    for entries in combined_map.values():
        for e in entries:
            for act in e['actual']:
                delays.append(act - e['scheduled'])

    if not delays:
        print("No delay data to plot.")
        return

    # Create interactive histogram with percent normalization
    fig = go.Figure(go.Histogram(
        x=delays,
        nbinsx=bins,
        histnorm='percent',
        name='Delay Distribution',
        marker_line_width=1,
        marker_line_color='black'
    ))

    fig.update_layout(
        title='Delay Distribution as Percentage',
        xaxis_title='Delay (s)',
        yaxis_title='Percentage of Total',
        bargap=0.1,
        template='plotly_white'
    )
    # Format y-axis ticks as percent with one decimal
    fig.update_yaxes(tickformat=".1f", ticksuffix="%")

    return fig

