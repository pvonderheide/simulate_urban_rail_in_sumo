import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Analysis of driving dynamics in SUMO""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    created by Paula von der Heide, TUBS, 2025

    If you use this code in your research or publications, please cite: "A SUMO-based study pf Urban Rail Operations on Frankfurts Corridor A", Paula von der Heide und Prof. Dr.-Ing. Lars Schnieder, 5.th International Railway Symposium Aachen, 2025

    not for commercial use
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## set up working environment""")
    return


@app.cell
def _():
    import xml.etree.ElementTree as ET
    import pandas as pd
    import plotly.express as px
    import os
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from dotenv import find_dotenv, load_dotenv, find_dotenv
    return ET, find_dotenv, load_dotenv, mo, norm, np, os, pd, plt, px


@app.cell
def _(find_dotenv, load_dotenv, os):
    # find my dotenv file
    notebook_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(notebook_dir)

    dotenv_path = find_dotenv(usecwd=True)  # <- important
    load_dotenv(dotenv_path, override=True) # override true otherwise once loaded variables will never update

    dotenv_path
    return


@app.cell
def _(os):
    # load path names and keys from dotenv
    scenario_dir = os.getenv("scenario_dir")
    return (scenario_dir,)


@app.cell
def _(os, scenario_dir):
    # switch working directory to scenario folder
    os.chdir(scenario_dir)
    print(os.getcwd())
    return


@app.cell
def _():
    # fcd data (floating car data = Position, Beschleunigung und Geschwindigkeit jedes Fahrzeuges in jedem Simulationsschritt)
    fcd_file = 'FrankfurtAsouth_out_fcdout.xml'
    return (fcd_file,)


@app.cell
def _():
    selected_vehicle = ["U2.1"]
    return (selected_vehicle,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## speed over time""")
    return


@app.cell
def _(ET, pd, px):
    def plot_speed_over_time(fcd_file, vehicle_ids):
        """
        Parse the SUMO FCD XML output and plot an interactive speed-over-time chart
        for the specified vehicle(s) using Plotly, converting speed to km/h,
        with all fonts set to 14 and no title.

        Parameters:
        - fcd_file (str): Path to the FCD output XML file.
        - vehicle_ids (list of str): List of vehicle IDs to plot.
        """
        # Parse XML and accumulate data
        tree = ET.parse(fcd_file)
        root = tree.getroot()
        records = []
        for timestep in root.findall('timestep'):
            t = float(timestep.get('time'))
            for veh in timestep.findall('vehicle'):
                vid = veh.get('id')
                if vid in vehicle_ids:
                    records.append({
                        'time': t,
                        'speed': float(veh.get('speed')),
                        'vehicle_id': vid
                    })

        # Build DataFrame and convert speed to km/h
        df = pd.DataFrame(records)
        df['speed_kmh'] = df['speed'] * 3.6

        # Create interactive line chart without title and with font size 14
        fig = px.line(
            df,
            x='time',
            y='speed_kmh',
            color='vehicle_id',
            labels={
                'time': 'Time [s]',
                'speed_kmh': 'Speed [km/h]',
                'vehicle_id': 'Vehicle ID'
            }
        )
        fig.update_layout(
            font=dict(size=14),
            hovermode='x unified'
        )
        fig.show()
    return (plot_speed_over_time,)


@app.cell
def _(fcd_file, plot_speed_over_time, selected_vehicle):
    plot_speed_over_time(fcd_file, selected_vehicle)
    return


@app.cell
def _(ET, pd, px):
    def plot_speed_interactive(fcd_file, vehicle_ids, start_time=None, end_time=None):
        """
        Parse the SUMO FCD XML output and plot an interactive speed-over-time chart
        for the specified vehicle(s) using Plotly, converting speed to km/h,
        with all fonts set to 14, axis title fonts set to 36, tick fonts set to 28,
        no title, optional time-frame filtering, a high-contrast Dark24 color palette,
        no legend, and minimal white space around the plot.

        Parameters:
        - fcd_file (str): Path to the FCD output XML file.
        - vehicle_ids (list of str): List of vehicle IDs to plot.
        - start_time (float, optional): Lower bound for time (inclusive) in seconds.
        - end_time (float, optional): Upper bound for time (inclusive) in seconds.
        """
        # Parse XML and accumulate data
        tree = ET.parse(fcd_file)
        root = tree.getroot()
        records = []
        for timestep in root.findall('timestep'):
            t = float(timestep.get('time'))
            for veh in timestep.findall('vehicle'):
                vid = veh.get('id')
                if vid in vehicle_ids:
                    records.append({'time': t, 'speed': float(veh.get('speed')), 'vehicle_id': vid})

        df = pd.DataFrame(records)

        # Filter by time frame if specified
        if start_time is not None:
            df = df[df['time'] >= start_time]
        if end_time is not None:
            df = df[df['time'] <= end_time]

        # Convert speed to km/h
        df['speed_kmh'] = df['speed'] * 3.6

        # Create interactive line chart with minimal margins
        fig = px.line(
            df,
            x='time',
            y='speed_kmh',
            color='vehicle_id',
            labels={'time': 'Time [s]', 'speed_kmh': 'Speed [km/h]', 'vehicle_id': 'Vehicle ID'},
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig.update_layout(
            font=dict(size=14),
            hovermode='x unified',
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title_font=dict(size=36), tickfont=dict(size=28)),
            yaxis=dict(title_font=dict(size=36), tickfont=dict(size=28))
        )
        fig.show()
    return (plot_speed_interactive,)


@app.cell
def _(fcd_file, plot_speed_interactive, selected_vehicle):
    # Plot speed between t=100s and t=200s:
    plot_speed_interactive(fcd_file, selected_vehicle, start_time=1030, end_time=1090)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## speed over distance""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""only works if distance is set in SUMO net file along the route!""")
    return


@app.cell
def _(ET, pd, px):
    def plot_speed_over_distance(fcd_file, vehicle_ids):
        """
        Parse the SUMO FCD XML output and plot an interactive speed-over-distance chart
        for the specified vehicle(s) using Plotly, converting speed to km/h,
        with all fonts set to 14 and no title.

        Notes:
        - Time is ignored.
        - Distance samples need not be equidistant.
        - If multiple samples share the same distance, the latest speed is used.
        """
        # Parse XML and accumulate distance/speed per selected vehicle(s)
        tree = ET.parse(fcd_file)
        root = tree.getroot()
        records = []
        for timestep in root.findall("timestep"):
            for veh in timestep.findall("vehicle"):
                vid = veh.get("id")
                if vid in vehicle_ids:
                    records.append({
                        "vehicle_id": vid,
                        "distance": float(veh.get("distance")),
                        "speed": float(veh.get("speed")),
                    })

        if not records:
            raise ValueError("No matching vehicle data found. Check vehicle_ids and file path.")

        # Build DataFrame and convert speed
        df = pd.DataFrame(records)
        df["speed_kmh"] = df["speed"] * 3.6
        print(df)

        # For each vehicle: keep the latest speed for duplicate distance values, then sort by distance
        cleaned = []
        for vid, sub in df.groupby("vehicle_id", sort=False):
            sub = sub.drop_duplicates(subset="distance", keep="last")
            sub = sub.sort_values("distance")
            cleaned.append(sub)
        df_clean = pd.concat(cleaned, ignore_index=True)

        # Plot (distance on x, speed on y), identical style to your time-based function
        fig = px.line(
            df_clean,
            x="distance",
            y="speed_kmh",
            color="vehicle_id",
            labels={
                "distance": "Distance [m]",
                "speed_kmh": "Speed [km/h]",
                "vehicle_id": "Vehicle ID",
            },
        )
        fig.update_layout(
            font=dict(size=14),
            hovermode="x unified",
        )
        fig.show()
    return (plot_speed_over_distance,)


@app.cell
def _(fcd_file, plot_speed_over_distance, selected_vehicle):
    plot_speed_over_distance(fcd_file, selected_vehicle)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## acceleration over time""")
    return


@app.cell
def _(ET, pd, px):
    def plot_acceleration(fcd_file, vehicle_ids):
        """
        Parse the SUMO FCD XML output and plot an interactive acceleration-over-time chart
        for the specified vehicle(s) using Plotly, with acceleration in m/s²,
        all fonts set to 14 and no title.
        """
        # Parse XML and accumulate data
        tree = ET.parse(fcd_file)
        root = tree.getroot()
        records = []
        for timestep in root.findall('timestep'):
            t = float(timestep.get('time'))
            for veh in timestep.findall('vehicle'):
                vid = veh.get('id')
                if vid in vehicle_ids:
                    records.append({
                        'time': t,
                        'speed': float(veh.get('speed')),
                        'vehicle_id': vid
                    })

        df = pd.DataFrame(records)
        # Compute acceleration (m/s²) per vehicle
        df = df.sort_values(['vehicle_id', 'time'])
        df['acceleration'] = df.groupby('vehicle_id')['speed'].diff() / df.groupby('vehicle_id')['time'].diff()

        fig = px.line(
            df,
            x='time',
            y='acceleration',
            color='vehicle_id',
            labels={
                'time': 'Time [s]',
                'acceleration': 'Acceleration [m/s²]',
                'vehicle_id': 'Vehicle ID'
            }
        )
        fig.update_layout(
            font=dict(size=14),
            hovermode='x unified'
        )
        fig.show()
    return (plot_acceleration,)


@app.cell
def _(fcd_file, plot_acceleration, selected_vehicle):
    plot_acceleration(fcd_file, selected_vehicle)
    return


@app.cell
def _(ET, pd, px):
    def plot_acceleration_interactive(fcd_file, vehicle_ids, start_time=None, end_time=None):
        """
        Parse the SUMO FCD XML output and plot an interactive acceleration-over-time chart
        for the specified vehicle(s) using Plotly, computing acceleration in m/s²,
        with all fonts set to 14, axis title fonts set to 36, tick fonts set to 28,
        no title, optional time-frame filtering, a high-contrast Dark24 color palette,
        no legend, and minimal white space around the plot.

        Parameters:
        - fcd_file (str): Path to the FCD output XML file.
        - vehicle_ids (list of str): List of vehicle IDs to plot.
        - start_time (float, optional): Lower bound for time (inclusive) in seconds.
        - end_time (float, optional): Upper bound for time (inclusive) in seconds.
        """
        # Parse XML and accumulate data
        tree = ET.parse(fcd_file)
        root = tree.getroot()
        records = []
        for timestep in root.findall('timestep'):
            t = float(timestep.get('time'))
            for veh in timestep.findall('vehicle'):
                vid = veh.get('id')
                if vid in vehicle_ids:
                    records.append({'time': t, 'speed': float(veh.get('speed')), 'vehicle_id': vid})

        df = pd.DataFrame(records)

        # Filter by time frame if specified
        if start_time is not None:
            df = df[df['time'] >= start_time]
        if end_time is not None:
            df = df[df['time'] <= end_time]

        # Compute acceleration (m/s²)
        df = df.sort_values(['vehicle_id', 'time'])
        df['acceleration'] = df.groupby('vehicle_id')['speed'].diff() / df.groupby('vehicle_id')['time'].diff()

        # Create interactive line chart with minimal margins
        fig = px.line(
            df,
            x='time',
            y='acceleration',
            color='vehicle_id',
            labels={'time': 'Time [s]', 'acceleration': 'Acceleration [m/s²]', 'vehicle_id': 'Vehicle ID'},
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig.update_layout(
            font=dict(size=14),
            hovermode='x unified',
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title_font=dict(size=36), tickfont=dict(size=28)),
            yaxis=dict(title_font=dict(size=36), tickfont=dict(size=28))
        )
        fig.show()
    return (plot_acceleration_interactive,)


@app.cell
def _(fcd_file, plot_acceleration_interactive, selected_vehicle):
    # Plot acceleration between t=50s and t=150s:
    plot_acceleration_interactive(fcd_file, selected_vehicle, start_time=1030, end_time=1090)
    return


@app.function
def calculate_avg_acceleration(fcd_file, vehicle_ids, start_time, end_time):
    """
    Parse the SUMO FCD XML, compute instantaneous acceleration (m/s²) per vehicle,
    and return the average acceleration across all specified vehicles
    between two time bounds.

    Parameters:
    - fcd_file     : str, path to FCD XML output file
    - vehicle_ids  : list of str, vehicle IDs to include
    - start_time   : float, lower time bound (inclusive) in seconds
    - end_time     : float, upper time bound (inclusive) in seconds

    Returns:
    - float: average acceleration in m/s² (0.0 if no data)
    """
    import xml.etree.ElementTree as ET
    import pandas as pd

    # Parse XML and collect speed-time records
    tree = ET.parse(fcd_file)
    root = tree.getroot()
    records = []
    for timestep in root.findall('timestep'):
        t = float(timestep.get('time'))
        for veh in timestep.findall('vehicle'):
            vid = veh.get('id')
            if vid in vehicle_ids:
                records.append({
                    'vehicle_id': vid,
                    'time': t,
                    'speed': float(veh.get('speed'))
                })

    # Build DataFrame
    df = pd.DataFrame(records)

    # Sort and compute acceleration per vehicle
    df = df.sort_values(['vehicle_id', 'time'])
    df['acceleration'] = (
        df.groupby('vehicle_id')['speed'].diff() /
        df.groupby('vehicle_id')['time'].diff()
    )

    # Filter within time bounds
    df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]

    # Drop NaNs and compute average
    acc_values = df['acceleration'].dropna()
    return acc_values.mean() if not acc_values.empty else 0.0


@app.cell
def _(fcd_file, selected_vehicle):
    avg_acc = calculate_avg_acceleration(
        fcd_file, selected_vehicle,
        start_time=1037,
        end_time=1049
    )
    print(f"Average acceleration: {avg_acc:.4f} m/s²")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## braking distance""")
    return


@app.cell
def _(ET, np, pd):
    def compute_braking_distances_refined_OLD(
        fcd_file,
        vehicle_id=None,
        stop_threshold_mps=0.5,
        lookback_window_m=300.0,
        dv_min_kmh=5.0,
        bump_tol_kmh=1.0,
        min_distance_m=20.0
    ):
        """
        Compute braking distances from a SUMO FCD XML file using a local-window method.

        Adds 2σ and 3σ variation magnitudes (not upper/lower bounds).
        """
        # --- normalize vehicle selection ---
        if vehicle_id is None:
            selected = None
        elif isinstance(vehicle_id, (list, tuple, set)):
            selected = {str(v).strip() for v in vehicle_id}
        else:
            selected = {str(vehicle_id).strip()}

        # --- parse XML ---
        tree = ET.parse(fcd_file)
        root = tree.getroot()

        records = []
        for ts in root.findall("timestep"):
            t = float(ts.get("time"))
            for v in ts.findall("vehicle"):
                vid = (v.get("id") or "").strip()
                if selected is None or vid in selected:
                    dist_attr = v.get("distance")
                    if dist_attr is None:
                        raise ValueError("FCD has no 'distance'. Run SUMO with --fcd-output.distance.")
                    records.append({
                        "vehicle_id": vid,
                        "time": t,
                        "distance": float(dist_attr),
                        "speed": float(v.get("speed"))
                    })

        if not records:
            raise ValueError(f"No samples found for vehicle_id(s) {selected} in {fcd_file}")

        df = pd.DataFrame(records).sort_values(["vehicle_id", "time"]).reset_index(drop=True)
        df = df.dropna(subset=["vehicle_id", "distance", "speed"])

        # --- helper: compute braking cycles for one vehicle ---
        def _cycles_for(sub):
            sub = sub[sub["distance"].diff().fillna(1) >= 0].reset_index(drop=True)
            spd_kmh = sub["speed"] * 3.6
            stopped = sub["speed"] <= stop_threshold_mps
            n = len(sub)
            i = 0
            rows = []

            while i < n:
                if stopped.iloc[i]:
                    stop_start = i
                    while i < n and stopped.iloc[i]:
                        i += 1
                    stop_dist = sub.at[stop_start, "distance"]

                    mask = (sub.index < stop_start) & (sub["distance"] >= stop_dist - lookback_window_m)
                    win = sub[mask]
                    if len(win) >= 3:
                        win_spd_kmh = win["speed"] * 3.6
                        loc_max_pos = [
                            j for j in range(1, len(win_spd_kmh) - 1)
                            if (win_spd_kmh.iloc[j] >= win_spd_kmh.iloc[j-1])
                            and (win_spd_kmh.iloc[j] >= win_spd_kmh.iloc[j+1])
                        ]
                        if loc_max_pos:
                            j_star = loc_max_pos[-1]
                            start_idx = win.index[j_star]
                        else:
                            start_idx = win_spd_kmh.idxmax()

                        start_dist = sub.at[start_idx, "distance"]
                        start_speed_kmh = float(spd_kmh.loc[start_idx])

                        post = sub.loc[start_idx:stop_start]
                        post_spd_kmh = post["speed"] * 3.6
                        drop_kmh = float(start_speed_kmh - post_spd_kmh.iloc[-1])

                        if (post_spd_kmh.max() <= start_speed_kmh + bump_tol_kmh) and (drop_kmh >= dv_min_kmh):
                            bd = float(stop_dist - start_dist)
                            if bd >= min_distance_m:
                                rows.append({
                                    "vehicle_id": sub["vehicle_id"].iloc[0],
                                    "start_time": float(sub.at[start_idx, "time"]),
                                    "stop_time": float(sub.at[stop_start, "time"]),
                                    "start_distance": float(start_dist),
                                    "stop_distance": float(stop_dist),
                                    "start_speed_kmh": float(start_speed_kmh),
                                    "stop_speed_kmh": float(post_spd_kmh.iloc[-1]),
                                    "braking_distance_m": bd
                                })
                else:
                    i += 1
            return pd.DataFrame(rows)

        # --- compute per vehicle ---
        out = []
        for vid, sub in df.groupby("vehicle_id", sort=False):
            sub = sub.sort_values("time").reset_index(drop=True)
            cycles = _cycles_for(sub)
            if not cycles.empty:
                out.append(cycles)

        cycles_df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
            columns=["vehicle_id","start_time","stop_time","start_distance","stop_distance",
                     "start_speed_kmh","stop_speed_kmh","braking_distance_m"]
        )

        # --- summary stats ---
        if cycles_df.empty:
            stats = {
                "count": 0,
                "mean_braking_distance_m": np.nan,
                "std_braking_distance_m": np.nan,
                "max_deviation_m": np.nan,
                "two_sigma_m": np.nan,
                "three_sigma_m": np.nan,
                "per_vehicle": {}
            }
            return cycles_df, stats

        per_vehicle = (
            cycles_df.groupby("vehicle_id")["braking_distance_m"]
            .agg(["count", "mean", "std"])
            .rename(columns={"mean": "mean_braking_distance_m", "std": "std_braking_distance_m"})
            .to_dict(orient="index")
        )

        mean_val = cycles_df["braking_distance_m"].mean()
        std_val = cycles_df["braking_distance_m"].std(ddof=1)
        max_dev = np.abs(cycles_df["braking_distance_m"] - mean_val).max()

        stats = {
            "count": int(cycles_df.shape[0]),
            "mean_braking_distance_m": float(mean_val),
            "std_braking_distance_m": float(std_val),
            "max_deviation_m": float(max_dev),
            "two_sigma_m": float(2 * std_val),
            "three_sigma_m": float(3 * std_val),
            "per_vehicle": per_vehicle
        }

        return cycles_df, stats
    return


@app.cell
def _(ET, np, pd):
    def compute_braking_distances_refined(
        fcd_file,
        vehicle_id=None,               # str | list | set | None (None => all vehicles)
        stop_threshold_mps=0.5,        # speed ≤ this => stop
        lookback_window_m=300.0,       # search braking start within last X meters before stop
        dv_min_kmh=5.0,                # require at least this drop from start to stop
        bump_tol_kmh=1.0,              # allow small speed bumps after start
        min_distance_m=20.0,           # discard short braking distances
        target_start_kmh=60.0,         # ← desired start speed (center)
        start_tol_kmh=5.0              # ← tolerance around target (±)
    ):
        """
        Compute braking distances from a SUMO FCD XML file using a local-window method.

        Additional filter: keep only braking cycles whose detected start speed is within
        [target_start_kmh - start_tol_kmh, target_start_kmh + start_tol_kmh] km/h.
        """
        # --- normalize vehicle selection ---
        if vehicle_id is None:
            selected = None
        elif isinstance(vehicle_id, (list, tuple, set)):
            selected = {str(v).strip() for v in vehicle_id}
        else:
            selected = {str(vehicle_id).strip()}

        # --- parse XML ---
        tree = ET.parse(fcd_file)
        root = tree.getroot()

        records = []
        for ts in root.findall("timestep"):
            t = float(ts.get("time"))
            for v in ts.findall("vehicle"):
                vid = (v.get("id") or "").strip()
                if selected is None or vid in selected:
                    dist_attr = v.get("distance")
                    if dist_attr is None:
                        raise ValueError("FCD has no 'distance'. Run SUMO with --fcd-output.distance.")
                    records.append({
                        "vehicle_id": vid,
                        "time": t,
                        "distance": float(dist_attr),
                        "speed": float(v.get("speed"))
                    })

        if not records:
            raise ValueError(f"No samples found for vehicle_id(s) {selected} in {fcd_file}")

        df = pd.DataFrame(records).sort_values(["vehicle_id", "time"]).reset_index(drop=True)
        df = df.dropna(subset=["vehicle_id", "distance", "speed"])

        # --- helper: compute braking cycles for one vehicle ---
        def _cycles_for(sub):
            sub = sub[sub["distance"].diff().fillna(1) >= 0].reset_index(drop=True)
            spd_kmh = sub["speed"] * 3.6
            stopped = sub["speed"] <= stop_threshold_mps
            n = len(sub)
            i = 0
            rows = []

            while i < n:
                if stopped.iloc[i]:
                    stop_start = i
                    while i < n and stopped.iloc[i]:
                        i += 1
                    stop_dist = sub.at[stop_start, "distance"]

                    # lookback window before stop
                    mask = (sub.index < stop_start) & (sub["distance"] >= stop_dist - lookback_window_m)
                    win = sub[mask]
                    if len(win) >= 3:
                        win_spd_kmh = win["speed"] * 3.6
                        # last local maximum in window (fallback: global max)
                        loc_max_pos = [
                            j for j in range(1, len(win_spd_kmh) - 1)
                            if (win_spd_kmh.iloc[j] >= win_spd_kmh.iloc[j-1])
                            and (win_spd_kmh.iloc[j] >= win_spd_kmh.iloc[j+1])
                        ]
                        if loc_max_pos:
                            j_star = loc_max_pos[-1]
                            start_idx = win.index[j_star]
                        else:
                            start_idx = win_spd_kmh.idxmax()

                        start_dist = sub.at[start_idx, "distance"]
                        start_speed_kmh = float(spd_kmh.loc[start_idx])

                        # --- NEW: keep only starts near target speed ---
                        if not (target_start_kmh - start_tol_kmh <= start_speed_kmh <= target_start_kmh + start_tol_kmh):
                            continue

                        post = sub.loc[start_idx:stop_start]
                        post_spd_kmh = post["speed"] * 3.6
                        drop_kmh = float(start_speed_kmh - post_spd_kmh.iloc[-1])

                        if (post_spd_kmh.max() <= start_speed_kmh + bump_tol_kmh) and (drop_kmh >= dv_min_kmh):
                            bd = float(stop_dist - start_dist)
                            if bd >= min_distance_m:
                                rows.append({
                                    "vehicle_id": sub["vehicle_id"].iloc[0],
                                    "start_time": float(sub.at[start_idx, "time"]),
                                    "stop_time": float(sub.at[stop_start, "time"]),
                                    "start_distance": float(start_dist),
                                    "stop_distance": float(stop_dist),
                                    "start_speed_kmh": float(start_speed_kmh),
                                    "stop_speed_kmh": float(post_spd_kmh.iloc[-1]),
                                    "braking_distance_m": bd
                                })
                else:
                    i += 1
            return pd.DataFrame(rows)

        # --- compute per vehicle ---
        out = []
        for vid, sub in df.groupby("vehicle_id", sort=False):
            sub = sub.sort_values("time").reset_index(drop=True)
            cycles = _cycles_for(sub)
            if not cycles.empty:
                out.append(cycles)

        cycles_df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
            columns=["vehicle_id","start_time","stop_time","start_distance","stop_distance",
                     "start_speed_kmh","stop_speed_kmh","braking_distance_m"]
        )

        # --- summary stats ---
        if cycles_df.empty:
            stats = {
                "count": 0,
                "mean_braking_distance_m": np.nan,
                "std_braking_distance_m": np.nan,
                "max_deviation_m": np.nan,
                "two_sigma_m": np.nan,
                "three_sigma_m": np.nan,
                "per_vehicle": {}
            }
            return cycles_df, stats

        per_vehicle = (
            cycles_df.groupby("vehicle_id")["braking_distance_m"]
            .agg(["count", "mean", "std"])
            .rename(columns={"mean": "mean_braking_distance_m", "std": "std_braking_distance_m"})
            .to_dict(orient="index")
        )

        mean_val = cycles_df["braking_distance_m"].mean()
        std_val = cycles_df["braking_distance_m"].std(ddof=1)
        max_dev = np.abs(cycles_df["braking_distance_m"] - mean_val).max()

        stats = {
            "count": int(cycles_df.shape[0]),
            "mean_braking_distance_m": float(mean_val),
            "std_braking_distance_m": float(std_val),
            "max_deviation_m": float(max_dev),
            "two_sigma_m": float(2 * std_val),
            "three_sigma_m": float(3 * std_val),
            "per_vehicle": per_vehicle
        }

        return cycles_df, stats
    return (compute_braking_distances_refined,)


@app.cell
def _(compute_braking_distances_refined, fcd_file):
    cycles_all, stats_all = compute_braking_distances_refined(
        fcd_file,
        vehicle_id=None,     # all vehicles
        min_distance_m=20.0  # discard < 20 m
    )
    return cycles_all, stats_all


@app.cell
def _(cycles_all):
    cycles_all
    return


@app.cell
def _(stats_all):
    stats_all
    return


@app.cell
def _(plt):
    def plot_braking_deviation(cycles_df, stats):
        """
        Plot braking distance deviation curve (distance vs deviation from mean).

        Parameters:
        - cycles_df: DataFrame from compute_braking_distances_refined()
        - stats: dict returned by compute_braking_distances_refined()
        """
        if cycles_df.empty:
            print("No braking cycles to plot.")
            return

        mean_val = stats["mean_braking_distance_m"]
        std_val = stats["std_braking_distance_m"]

        # Compute deviations
        deviations = cycles_df["braking_distance_m"] - mean_val

        plt.figure(figsize=(10,5))
        plt.plot(deviations.values, marker='o', linestyle='-', linewidth=1.5, label='Deviation')
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.axhline( std_val, color='orange', linestyle=':', linewidth=1.5, label='+1σ')
        plt.axhline(-std_val, color='orange', linestyle=':', linewidth=1.5)
        plt.axhline( 2*std_val, color='red', linestyle='--', linewidth=1.2, label='+2σ')
        plt.axhline(-2*std_val, color='red', linestyle='--', linewidth=1.2)
        plt.axhline( 3*std_val, color='purple', linestyle='-.', linewidth=1.2, label='+3σ')
        plt.axhline(-3*std_val, color='purple', linestyle='-.', linewidth=1.2)

        plt.title("Braking Distance Deviations from Mean")
        plt.xlabel("Braking Cycle Index")
        plt.ylabel("Deviation from Mean [m]")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    return (plot_braking_deviation,)


@app.cell
def _(compute_braking_distances_refined, fcd_file, plot_braking_deviation):
    cycles, stats = compute_braking_distances_refined(
        fcd_file,
        vehicle_id=None,
        target_start_kmh=60,
        start_tol_kmh=5
    )

    plot_braking_deviation(cycles, stats)
    return cycles, stats


@app.cell
def _(norm, np, plt):
    def plot_braking_distribution(
        cycles_df,
        stats,
        target_start_kmh=None,
        start_tol_kmh=None
    ):
        """
        Plot a clean normal distribution of braking distances with labeled mean (μ)
        and ±1σ, ±2σ, ±3σ boundaries. Console output lists full σ ranges and widths.
        """

        if cycles_df.empty:
            print("No data to plot.")
            return

        # === Statistik ===
        mean_val = stats["mean_braking_distance_m"]
        std_val  = stats["std_braking_distance_m"]

        sigma_levels = [1, 2, 3]
        limits = {f"{i}σ": (mean_val - i*std_val, mean_val + i*std_val) for i in sigma_levels}

        # === Normalverteilung ===
        x = np.linspace(mean_val - 4*std_val, mean_val + 4*std_val, 400)
        y = norm.pdf(x, mean_val, std_val)

        plt.figure(figsize=(11, 5))
        plt.plot(x, y, color='black', linewidth=2)

        # Schattierte σ-Bereiche
        colors = ['#cce5ff', '#99ccff', '#6699ff']
        for i, c in zip(sigma_levels, colors):
            low, high = limits[f"{i}σ"]
            plt.fill_between(x, 0, y, where=(x >= low) & (x <= high), color=c, alpha=0.7)

        # Linien + Labels auf gleicher Höhe
        y_label = plt.ylim()[1] * 0.85
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2)
        plt.text(mean_val, y_label, f"μ = {mean_val:.2f} m",
                 color='red', ha='center', va='bottom', fontsize=12, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

        for i, color in zip(sigma_levels, colors):
            low, high = limits[f"{i}σ"]
            plt.axvline(low,  color='gray', linestyle='--', linewidth=1)
            plt.axvline(high, color='gray', linestyle='--', linewidth=1)
            for pos, sign in [(low, f"-{i}σ"), (high, f"+{i}σ")]:
                plt.text(pos, y_label, f"{sign}\n{pos:.2f} m",
                         ha='center', va='bottom', fontsize=10, color='black',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

        # Titel
        if target_start_kmh is not None:
            title = f"Verteilung der Bremswege (Startgeschwindigkeit ≈ {target_start_kmh} ± {start_tol_kmh} km/h)"
        else:
            title = "Verteilung der Bremswege"
        plt.title(title, fontsize=14, fontweight="bold")

        plt.xlabel("Bremsweg [m]", fontsize=12)
        plt.ylabel("Wahrscheinlichkeitsdichte", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # === Konsole ===
        print("Konkrete Bremsweggrenzen und Spannweiten:")
        for k, (low, high) in limits.items():
            span = high - low
            print(f"  {k}:  {low:.2f} m  –  {high:.2f} m  =  {span:.2f} m")
    return (plot_braking_distribution,)


@app.cell
def _(cycles, plot_braking_distribution, stats):
    plot_braking_distribution(
        cycles, stats,
        target_start_kmh=60,
        start_tol_kmh=5
    )
    return


if __name__ == "__main__":
    app.run()
