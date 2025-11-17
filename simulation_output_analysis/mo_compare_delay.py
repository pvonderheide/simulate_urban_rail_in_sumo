import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Analysis for IRSA scenarios conventional vs CBTC""")
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
    mo.md(
        r"""
    - This script accesses the simulation data created by a SUMO simulation for different delay values as created by mo_add_fixed_delay_multipl.py
    - We compare the impact of the automated vs. non-automated reversal
    - We compare the impact of fixed vs. moving block
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Set which simulation and vehicles to evaluate**""")
    return


@app.cell
def _(dir, mo, os):
    os.chdir(dir)
    mo.output.append(os.getcwd())
    return


@app.cell
def _():
    # which vehicles to analyze
    delayed_vehicle = "U2.1"
    successor_1 = "U3.1"
    successor_2 = "U1.2"
    #relevant_vehicles = [delayed_vehicle, successor_1, successor_2]
    relevant_vehicles = [delayed_vehicle, successor_1, successor_2, "U8.1", "U2.2"]
    #relevant_vehicles = [delayed_vehicle, successor_1, successor_2, "U8.1","U2.2", "U1.3"]
    return (relevant_vehicles,)


@app.cell
def _():
    delay = 300
    return (delay,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## getting the data""")
    return


@app.cell
def _(find_dotenv, load_dotenv, os):
    # find my dotenv file
    notebook_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(notebook_dir)

    dotenv_path = find_dotenv(usecwd=True)  # <- important
    load_dotenv(dotenv_path, override=True) # override true otherwise once loaded variables will never update

    dotenv_path

    dir = os.getenv("IRSA_main_folder")
    path_to_sources=os.getenv("SUMO_src")
    return dir, path_to_sources


@app.cell
def _():
    import marimo as mo
    import os
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as px
    import math
    import re
    import sys
    from dotenv import find_dotenv, load_dotenv, find_dotenv
    return find_dotenv, go, load_dotenv, math, mo, os, plt, px, re, sys


@app.cell
def _(path_to_sources, sys):
    # import functions from other source code

    sys.path.append(path_to_sources)
    #sys.path.append(r"C:\Users\von der Heide\Documents\gitlab\src\SUMO-src")
    import simulation_output_analysis.delay_visualization as vd
    import simulation_output_analysis.delay_calc as cd
    return (cd,)


@app.cell
def _():
    #set path prefix to access scenario folders!

    #path_prefix_conv = r"scenario_fixedblock"
    path_prefix_conv = r"scenario_fixedblock3_signalsfixed"
    #path_prefix_CBTC = r"scenario_movingblock2"
    path_prefix_CBTC = r"scenario_movingblock3_signalsfixed"
    return path_prefix_CBTC, path_prefix_conv


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**defining_scenarios**""")
    return


@app.cell
def _(path_prefix_CBTC, path_prefix_conv):
    scenarios_all = {
        "120s delay fixed block": {
            "routes_file": path_prefix_conv+"/FrankfurtAsouth_flows_stops.rou.xml",
            "stop_file": path_prefix_conv+"/FrankfurtAsouth_trainStops.add.xml",
            "stop_output_file": path_prefix_conv+"fixedDelay120/FrankfurtAsouth_out_stop-output.xml",
            "fcd_output_file": path_prefix_conv+"/fixedDelay120/FrankfurtAsouth_out_fcdout.xml",
        },
        "180s delay fixed block": {
            "routes_file": path_prefix_conv+"/FrankfurtAsouth_flows_stops.rou.xml",
            "stop_file": path_prefix_conv+"/FrankfurtAsouth_trainStops.add.xml",
            "stop_output_file": path_prefix_conv+"/fixedDelay180/FrankfurtAsouth_out_stop-output.xml",
            "fcd_output_file": path_prefix_conv+"/fixedDelay180/FrankfurtAsouth_out_fcdout.xml",
        },
        "300s delay fixed block": {
            "routes_file": path_prefix_conv+"/FrankfurtAsouth_flows_stops.rou.xml",
            "stop_file": path_prefix_conv+"/FrankfurtAsouth_trainStops.add.xml",
            "stop_output_file": path_prefix_conv+"/fixedDelay300/FrankfurtAsouth_out_stop-output.xml",
            "fcd_output_file": path_prefix_conv+"/fixedDelay300/FrankfurtAsouth_out_fcdout.xml",
        },
        "600s delay fixed block": {
            "routes_file": path_prefix_conv+"/FrankfurtAsouth_flows_stops.rou.xml",
            "stop_file": path_prefix_conv+"/FrankfurtAsouth_trainStops.add.xml",
            "stop_output_file": path_prefix_conv+"/fixedDelay600/FrankfurtAsouth_out_stop-output.xml",
            "fcd_output_file": path_prefix_conv+"/fixedDelay600/FrankfurtAsouth_out_fcdout.xml",
        },
        "120s delay moving block": {
            "routes_file": path_prefix_CBTC+"/FrankfurtAsouth_flows_stops.rou.xml",
            "stop_file": path_prefix_CBTC+"/FrankfurtAsouth_trainStops.add.xml",
            "stop_output_file": path_prefix_CBTC+"/fixedDelay120/FrankfurtAsouth_out_stop-output.xml",
            "fcd_output_file": path_prefix_CBTC+"/fixedDelay120/FrankfurtAsouth_out_fcdout.xml",
        },
        "180s delay moving block": {
            "routes_file": path_prefix_CBTC+"/FrankfurtAsouth_flows_stops.rou.xml",
            "stop_file": path_prefix_CBTC+"/FrankfurtAsouth_trainStops.add.xml",
            "stop_output_file": path_prefix_CBTC+"/fixedDelay180/FrankfurtAsouth_out_stop-output.xml",
            "fcd_output_file": path_prefix_CBTC+"/fixedDelay180/FrankfurtAsouth_out_fcdout.xml",
        },
        "300s delay moving block": {
            "routes_file": path_prefix_CBTC+"/FrankfurtAsouth_flows_stops.rou.xml",
            "stop_file": path_prefix_CBTC+"/FrankfurtAsouth_trainStops.add.xml",
            "stop_output_file": path_prefix_CBTC+"/fixedDelay300/FrankfurtAsouth_out_stop-output.xml",
            "fcd_output_file": path_prefix_CBTC+"/fixedDelay300/FrankfurtAsouth_out_fcdout.xml",
        },
        "600s delay moving block": {
            "routes_file": path_prefix_CBTC+"/FrankfurtAsouth_flows_stops.rou.xml",
            "stop_file": path_prefix_CBTC+"/FrankfurtAsouth_trainStops.add.xml",
            "stop_output_file": path_prefix_CBTC+"/fixedDelay600/FrankfurtAsouth_out_stop-output.xml",
            "fcd_output_file": path_prefix_CBTC+"/fixedDelay600/FrankfurtAsouth_out_fcdout.xml",
        },
    }
    return (scenarios_all,)


@app.cell
def _(cd, delays, relevant_vehicles, scheduled_stops):
    # === Load and align actual times per scenario for different delays ===
    blocks = [
        ("fixed block",  "scenario_fixedblock3_signalsfixed"),
        ("moving block", "scenario_movingblock3_signalsfixed")
    ]

    # ------------- build the aligned results dict -------------
    results_per_vehicle_all = { _vid: {} for _vid in relevant_vehicles }

    for _d in delays:
        for _block_name, _folder in blocks:
            _key      = f"{_d}s delay {_block_name}"
            _xml_path = f"{_folder}/fixedDelay{_d}/FrankfurtAsouth_out_stop-output.xml"
            _actual_map = cd.parse_actuals(_xml_path, relevant_vehicles)

            for _vid in relevant_vehicles:
                # build a stop→times lookup
                _actual_list = _actual_map[_vid]
                _actual_dict = { _stop: _times for _stop, _times in _actual_list }

                # align to the scheduled stops exactly
                _aligned = []
                for _stop in scheduled_stops[_vid]:
                    _times = _actual_dict.get(_stop, [])
                    if isinstance(_times, (list, tuple)):
                        _aligned.append(_times[0] if _times else None)
                    elif _times is None:
                        _aligned.append(None)
                    else:
                        _aligned.append(_times)

                results_per_vehicle_all[_vid][_key] = _aligned
    return (results_per_vehicle_all,)


@app.cell
def _(delay):
    foldername_conv="scenario_fixedblock3_signalsfixed"
    fodlername_CBTC="scenario_movingblock3_signalsfixed"

    scenarios1 = {
        f"{delay}s delay fixed block": {
            "routes_file": f"{foldername_conv}/FrankfurtAsouth_flows_stops.rou.xml",
            "stop_file": f"{foldername_conv}/FrankfurtAsouth_trainStops.add.xml",
            "stop_output_file": f"{foldername_conv}/fixedDelay{delay}/FrankfurtAsouth_out_stop-output.xml",
            "fcd_output_file": f"{foldername_conv}/fixedDelay{delay}/FrankfurtAsouth_out_fcdout.xml",
        },
        f"{delay}s delay moving block": {
            "routes_file":      f"{fodlername_CBTC}/FrankfurtAsouth_flows_stops.rou.xml",
            "stop_file": "{fodlername_CBTC}/FrankfurtAsouth_trainStops.add.xml",
            "stop_output_file": f"{fodlername_CBTC}/fixedDelay{delay}/FrankfurtAsouth_out_stop-output.xml",
            "fcd_output_file": f"{fodlername_CBTC}/fixedDelay{delay}/FrankfurtAsouth_out_fcdout.xml",
        },   
    }
    return (scenarios1,)


@app.cell
def _(cd, relevant_vehicles, scenarios_all):
    # === 1) Load the schedule ===
    first = next(iter(scenarios_all.values()))
    sched_map = cd.create_schedule_from_route_definition(first["routes_file"], relevant_vehicles)

    # Extract stop names and scheduled times per vehicle
    scheduled_stops = {}
    scheduled_times = {}
    for _vid in relevant_vehicles:
        stop_tuples = sched_map.get(_vid, [])
        # stop_tuples: list of (stop_name, scheduled_time)
        scheduled_stops[_vid] = [stop for stop, _ in stop_tuples]
        scheduled_times[_vid] = [time for _, time in stop_tuples]
    return scheduled_stops, scheduled_times


@app.cell
def _(cd, relevant_vehicles, scenarios1, scheduled_stops):
    # === 2b) Load and align actual times per scenario ===
    results_per_vehicle1 = {_vid: {} for _vid in relevant_vehicles}

    for label, paths in scenarios1.items():
        actual_map = cd.parse_actuals(paths["stop_output_file"], relevant_vehicles)
        for _vid in relevant_vehicles:
            # actual_map[_vid]: list of (stop_name, actual_time or list-of-times)
            # Build a lookup dict: stop_name -> actual_time_list
            actual_list = actual_map.get(_vid, [])
            actual_dict = {stop: times for stop, times in actual_list}

            # Align actuals to the scheduled stops order
            aligned_actuals = []
            for stop in scheduled_stops[_vid]:
                times = actual_dict.get(stop, [])
                # times might be a float or list-of-floats depending on cd.parse_actuals
                if isinstance(times, (list, tuple)):
                    aligned_actuals.append(times[0] if times else None)
                elif times is None or times == []:
                    aligned_actuals.append(None)
                else:
                    aligned_actuals.append(times)

            results_per_vehicle1[_vid][label] = aligned_actuals
    return (results_per_vehicle1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Space Time Diagram for one vehicle and different delays""")
    return


@app.cell
def _():
    delays = [120, 180, 300, 600, 900]   # you can only include values which have been simulated! folder for these delay values must exist
    return (delays,)


@app.cell
def _(plt, re):
    def plot_time_space_diagram(vehicle_id,
                                scheduled_stops,
                                scheduled_times,
                                results_per_vehicle,
                                fontsize=12,
                                legend_outside=None,
                                figsize=(10, 6),
                                auto_expand_ratio=1.3):
        """
        Plots a time–space diagram with:
          - scheduled timetable as solid black
          - for each delay group (e.g. “120 s”):
              * conventional runs as solid lines
              * CBTC runs as dashed lines
              * both share the same color
        """
        # Decide legend placement
        if legend_outside is None:
            legend_outside = fontsize > 12
            expanded = True
        else:
            expanded = False

        # Possibly expand width
        width, height = figsize
        if legend_outside and expanded:
            width *= auto_expand_ratio

        # Prepare color mapping by delay
        cmap = plt.get_cmap('tab10')
        all_colors = cmap.colors

        # Extract all delay values (as strings) from labels
        labels = results_per_vehicle[vehicle_id].keys()
        delays = []
        for lbl in labels:
            m = re.search(r'(\d+)s', lbl.lower())
            if m:
                delays.append(m.group(1))
        unique_delays = sorted(set(delays), key=int)

        # Map each delay to a distinct color
        color_map = {
            delay: all_colors[i % len(all_colors)]
            for i, delay in enumerate(unique_delays)
        }

        # Start plotting
        fig, ax = plt.subplots(figsize=(width, height))

        # Base data
        y_coords    = list(range(len(scheduled_stops[vehicle_id])))
        stop_labels = scheduled_stops[vehicle_id]
        times_sched = scheduled_times[vehicle_id]

        # 1) Scheduled timetable: solid black
        ax.plot(times_sched, y_coords,
                color='black', linewidth=2,
                label='Scheduled timetable')

        # 2) Actual runs, color by delay, linestyle by mode
        for lbl, actual_times in results_per_vehicle[vehicle_id].items():
            lower = lbl.lower()
            # extract delay
            m = re.search(r'(\d+)s', lower)
            delay = m.group(1) if m else None
            color = color_map.get(delay, 'gray')

            # choose linestyle and mode
            if 'fixed' in lower:
                ls = '-'
                mode = 'conventional'
            elif 'moving' in lower:
                ls = '--'
                mode = 'CBTC'
            else:
                ls = '--'
                mode = lbl

            # build legend label
            delay_str = f"{delay} s delay" if delay else ""
            legend_label = f"Actual ({delay_str}, {mode})"

            ax.plot(actual_times, y_coords,
                    color=color,
                    linestyle=ls,
                    linewidth=2.5,
                    marker='x',
                    label=legend_label)

        # 3) Format axes
        ax.invert_yaxis()
        ax.set_yticks(y_coords)
        ax.set_yticklabels(stop_labels, fontsize=fontsize)
        ax.set_xlabel('Time (s)', fontsize=fontsize)
        ax.set_ylabel('Stops', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.grid(True, linestyle=':')

        # 4) Legend placement
        if legend_outside:
            plt.subplots_adjust(right=0.75)
            ax.legend(loc='upper left',
                      bbox_to_anchor=(1.02, 1),
                      borderaxespad=0.,
                      fontsize=fontsize)
        else:
            ax.legend(loc='upper right', fontsize=fontsize)

        plt.tight_layout()
        plt.show()
    return


@app.cell
def _(plt, re):
    def plot_time_space_diagram_reversingLine(
        vehicle_id,
        scheduled_stops,
        scheduled_times,
        results_per_vehicle,
        fontsize=12,
        legend_outside=None,
        figsize=(10, 6),
        auto_expand_ratio=1.3,
        # Reversing marker configuration
        reversing_pair=("Südbahnhof_0", "Südbahnhof_1"),
        reversing_label="Reversing",
        reversing_label_pos="right",  # 'left' or 'right'
    ):
        """
        Plots a time–space diagram with:
          - scheduled timetable as solid black (drawn last so it’s always visible)
          - for each delay group (e.g. “120 s”):
              * conventional runs as solid lines
              * CBTC runs as dashed lines
              * both share the same color
          - marks a reversing area as a horizontal dashed black line between two stops
            and labels it on the chosen side ('left' or 'right')
        """

        # Decide legend placement
        if legend_outside is None:
            legend_outside = fontsize > 12
            expanded = True
        else:
            expanded = False

        # Possibly expand width
        width, height = figsize
        if legend_outside and expanded:
            width *= auto_expand_ratio

        # Prepare color mapping by delay
        cmap = plt.get_cmap('tab10')
        all_colors = cmap.colors

        # Extract all delay values (as strings) from labels
        labels = results_per_vehicle[vehicle_id].keys()
        delays = []
        for lbl in labels:
            m = re.search(r'(\d+)s', lbl.lower())
            if m:
                delays.append(m.group(1))
        unique_delays = sorted(set(delays), key=int)

        # Map each delay to a distinct color
        color_map = {
            delay: all_colors[i % len(all_colors)]
            for i, delay in enumerate(unique_delays)
        }

        # Start plotting
        fig, ax = plt.subplots(figsize=(width, height))

        # Base data
        y_coords    = list(range(len(scheduled_stops[vehicle_id])))
        stop_labels = scheduled_stops[vehicle_id]
        times_sched = scheduled_times[vehicle_id]

        # 1) Actual runs, color by delay, linestyle by mode
        for lbl, actual_times in results_per_vehicle[vehicle_id].items():
            lower = lbl.lower()
            # extract delay
            m = re.search(r'(\d+)s', lower)
            delay = m.group(1) if m else None
            color = color_map.get(delay, 'gray')

            # choose linestyle and mode
            if 'fixed' in lower:
                ls = '-'
                mode = 'conventional'
            elif 'moving' in lower:
                ls = '--'
                mode = 'CBTC'
            else:
                ls = '--'
                mode = lbl

            # build legend label
            delay_str = f"{delay} s delay" if delay else ""
            legend_label = f"Actual ({delay_str}, {mode})"

            ax.plot(actual_times, y_coords,
                    color=color,
                    linestyle=ls,
                    linewidth=2.5,
                    marker='x',
                    label=legend_label)

        # 2) Scheduled timetable: solid black (drawn last → always visible)
        ax.plot(times_sched, y_coords,
                color='black', linewidth=2.5,
                label='Scheduled timetable')

        # --- Mark reversing area between reversing_pair[0] and reversing_pair[1] ---
        if reversing_pair is not None:
            a, b = reversing_pair
            if a in stop_labels and b in stop_labels:
                y0 = stop_labels.index(a)
                y1 = stop_labels.index(b)
                y_rev = (y0 + y1) / 2  # halfway between the two stops

                # Draw horizontal dashed black line across the full time range
                xmin, xmax = ax.get_xlim()
                ax.hlines(y=y_rev, xmin=xmin, xmax=xmax,
                          colors='black', linestyles='dashed', linewidth=1.8)

                # --- Label placement ---
                if reversing_label_pos.lower() == "left":
                    x_text = xmin + 0.02 * (xmax - xmin)
                    ha = "left"
                else:  # default = right
                    x_text = xmax - 0.02 * (xmax - xmin)
                    ha = "right"

                ax.text(x_text, y_rev, reversing_label,
                        color='black', fontsize=fontsize,
                        va='center', ha=ha,
                        backgroundcolor='white')

                # Keep x-limits unchanged
                ax.set_xlim(xmin, xmax)

        # 3) Format axes
        ax.invert_yaxis()
        ax.set_yticks(y_coords)
        ax.set_yticklabels(stop_labels, fontsize=fontsize)
        ax.set_xlabel('Time (s)', fontsize=fontsize)
        ax.set_ylabel('Stops', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.grid(True, linestyle=':')

        # 4) Legend placement
        if legend_outside:
            plt.subplots_adjust(right=0.75)
            ax.legend(loc='upper left',
                      bbox_to_anchor=(1.02, 1),
                      borderaxespad=0.,
                      fontsize=fontsize)
        else:
            ax.legend(loc='upper right', fontsize=fontsize)

        plt.tight_layout()
        plt.show()
    return (plot_time_space_diagram_reversingLine,)


@app.cell
def _(
    plot_time_space_diagram_reversingLine,
    results_per_vehicle_all,
    scheduled_stops,
    scheduled_times,
):
    plot_time_space_diagram_reversingLine("U2.1", scheduled_stops, scheduled_times, results_per_vehicle_all,
                            fontsize=16,
                            legend_outside=True,
                            figsize=(14, 6),
                            reversing_pair=("Südbahnhof_0", "Südbahnhof_1"),
                            reversing_label="Reversing",
                            reversing_label_pos="left")
    return


@app.cell
def _(go):
    def plot_interactive_time_space_diagram(_vid,
                                            scheduled_stops,
                                            scheduled_times,
                                            results_per_vehicle):
        stops = scheduled_stops[_vid]
        times_sched = scheduled_times[_vid]
        y_idx = list(range(len(stops)))

        fig = go.Figure()

        # Scheduled timetable (solid black)
        fig.add_trace(go.Scatter(
            x=times_sched,
            y=y_idx,
            mode='lines+markers',
            name='Scheduled timetable',
            line=dict(color='black', width=2),
            hovertemplate='Stop: %{text}<br>Time: %{x}s',
            text=stops
        ))

        # Each rescheduled run (dashed red)
        for label, actuals in results_per_vehicle[_vid].items():
            fig.add_trace(go.Scatter(
                x=actuals,
                y=y_idx,
                mode='lines+markers',
                name=f'Rescheduled ({label})',
                line=dict(dash='dash'),
                hovertemplate='Stop: %{text}<br>Time: %{x}s',
                text=stops
            ))

        fig.update_layout(
            title=f"Interactive Time–Space Diagram for Vehicle {_vid}",
            xaxis=dict(title='Time (s)'),
            yaxis=dict(title='Stops',
                       tickmode='array',
                       tickvals=y_idx,
                       ticktext=stops,
                       autorange='reversed'),
            legend=dict(title=''),
            hovermode='closest',
            margin=dict(l=150, r=50, t=80, b=50)
        )

        fig.show()
    return


@app.cell
def _():
    #plot_interactive_time_space_diagram("U2.1",scheduled_stops,scheduled_times,results_per_vehicle_all)
    return


@app.cell
def _(go, px):
    def plot_interactive_space_time_diagram_colored(_vid,
                                                    scheduled_stops,
                                                    scheduled_times,
                                                    results_per_vehicle):
        stops       = scheduled_stops[_vid]
        times_sched = scheduled_times[_vid]
        palette     = px.colors.qualitative.Plotly  # e.g. ['#636EFA','#EF553B',...]

        fig = go.Figure()

        # 1) Scheduled timetable (solid black)
        fig.add_trace(go.Scatter(
            x=stops,
            y=times_sched,
            mode='lines+markers',
            name='Scheduled timetable',
            line=dict(color='black', width=2),
            hovertemplate='Stop: %{x}<br>Scheduled: %{y:.1f}s'
        ))

        # 2) Each rescheduled run with its own color
        for idx, (label, actuals) in enumerate(results_per_vehicle[_vid].items()):
            color = palette[idx % len(palette)]

            # main actual line
            fig.add_trace(go.Scatter(
                x=stops,
                y=actuals,
                mode='lines+markers',
                name=f'Rescheduled ({label})',
                line=dict(color=color, dash='dash', width=2),
                marker=dict(color=color),
                hovertemplate='Stop: %{x}<br>Actual: %{y:.1f}s'
            ))

            # recovered points (actual < scheduled), same color
            rec_x, rec_y = [], []
            for stop, actual, sched in zip(stops, actuals, times_sched):
                if actual is not None and actual < sched:
                    rec_x.append(stop)
                    rec_y.append(actual)

            if rec_x:
                fig.add_trace(go.Scatter(
                    x=rec_x,
                    y=rec_y,
                    mode='markers',
                    name=f'Recovered ({label})',
                    marker=dict(symbol='triangle-up', size=10, color=color),
                    hovertemplate='Stop: %{x}<br>Recovered: %{y:.1f}s'
                ))

        # 3) Layout with categorical stops and inverted y-axis
        fig.update_layout(
            title=f"Interactive Space-Time Diagram for Vehicle {_vid}",
            xaxis=dict(
                title='Stop',
                tickangle=45,
                categoryorder='array',
                categoryarray=stops
            ),
            yaxis=dict(
                title='Time (s)',
                autorange='reversed'
            ),
            legend=dict(title=''),
            hovermode='closest',
            margin=dict(l=80, r=50, t=80, b=150)
        )

        fig.show()
    return


@app.cell
def _():
    # Usage: only for U2.1
    #plot_interactive_space_time_diagram_colored("U2.1",scheduled_stops,scheduled_times,results_per_vehicle_all)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Space Time Diagram showing Impact on successor and 2nd successor""")
    return


@app.cell
def _(
    relevant_vehicles,
    results_per_vehicle1,
    scenarios1,
    scheduled_stops,
    scheduled_times,
):
    # ── Prepare empty trimmed dicts ──
    scheduled_stops_trimmed  = {}
    scheduled_times_trimmed  = {}
    results_per_vehicle1_trimmed = {vid: {} for vid in relevant_vehicles}

    # ── Trim all data to stops up through "Südbahnhof_0" ──
    cut_stop = "Südbahnhof_0"
    for _vid in relevant_vehicles:
        # find the index of the cut stop (inclusive)
        _idx = scheduled_stops[_vid].index(cut_stop) + 1

        # slice into the new dicts
        scheduled_stops_trimmed[_vid] = scheduled_stops[_vid][:_idx]
        scheduled_times_trimmed[_vid] = scheduled_times[_vid][:_idx]

        for _scenario_name in scenarios1:
            results_per_vehicle1_trimmed[_vid][_scenario_name] = (
                results_per_vehicle1[_vid][_scenario_name][:_idx]
            )
    return


@app.cell
def _(plt):
    def plot_time_space(
        vehicles, delay,
        stops_dict, times_dict, results_dict,
        title_suffix="",
        base_fontsize=12,
        colormap_name="tab10",
    ):
        """
        Plot a time–space diagram comparing fixed vs moving block for given vehicles.

        - Scheduled timetable: solid black
        - Fixed block: solid line, unique vehicle color
        - Moving block: dashed line, same color
        """

        # Build the scenario keys from the delay
        fixed_key  = f"{delay}s delay fixed block"
        moving_key = f"{delay}s delay moving block"

        # Prepare colormap for automatic distinct colors
        cmap = plt.get_cmap(colormap_name)
        n_colors = cmap.N
        vehicle_colors = {
            vid: cmap(i % n_colors)
            for i, vid in enumerate(sorted(vehicles))
        }

        # Font size setup (self-contained)
        plt.rcParams.update({
            "font.size": base_fontsize,
            "axes.titlesize": base_fontsize + 4,
            "axes.labelsize": base_fontsize + 2,
            "xtick.labelsize": base_fontsize,
            "ytick.labelsize": base_fontsize,
            "legend.fontsize": base_fontsize,
        })

        fig, ax = plt.subplots(figsize=(10, 6))

        for vid in vehicles:
            y = list(range(len(stops_dict[vid])))
            color = vehicle_colors[vid]

            # Scheduled: black solid
            ax.plot(
                times_dict[vid], y,
                color='black', linestyle='-', linewidth=2,
                label=f"Sched {vid}"
            )

            # Fixed block: solid, vehicle color
            if fixed_key in results_dict[vid]:
                ax.plot(
                    results_dict[vid][fixed_key], y,
                    color=color, linestyle='-', marker='x',
                    linewidth=2, label=f"{vid} fixed"
                )

            # Moving block: dashed, vehicle color
            if moving_key in results_dict[vid]:
                ax.plot(
                    results_dict[vid][moving_key], y,
                    color=color, linestyle='--', marker='x',
                    linewidth=2, label=f"{vid} moving"
                )

        # Axis formatting
        ax.invert_yaxis()
        ax.set_yticks(y)
        ax.set_yticklabels(stops_dict[vehicles[0]])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Stop")

        # Title
        suffix = f" — {title_suffix}" if title_suffix else ""
        ax.set_title(f"Time–Space Diagram for {', '.join(vehicles)}, — {delay}s of U2.1{suffix}")

        # Legend (combine duplicates, place outside)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(), by_label.keys(),
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            ncol=1
        )

        # Grid and layout
        ax.grid(True, linestyle=':')
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        plt.show()
    return


@app.cell
def _():
    # Example usage with full data:
    #plot_time_space(relevant_vehicles, delay, scheduled_stops, scheduled_times, results_per_vehicle1, title_suffix="full")

    # Example usage with trimmed data:
    #plot_time_space(relevant_vehicles, delay, scheduled_stops_trimmed, scheduled_times_trimmed, results_per_vehicle1_trimmed, title_suffix="trimmed")
    return


@app.cell
def _(go, px):
    def plot_interactive_time_space_dynamic(vehicles, delay, cut_stop,
                                            scheduled_stops, scheduled_times,
                                            results_per_vehicle):
        """
        Interactive time–space diagram that adapts to any number of vehicles.

        vehicles             : list of vehicle IDs
        delay                : integer delay in seconds
        cut_stop             : the last stop to include
        scheduled_stops      : dict {vid: [stop1, stop2, ...]}
        scheduled_times      : dict {vid: [time1, time2, ...]}
        results_per_vehicle  : dict {vid: {scenario_key: [time, ...], ...}, ...}
        """
        # Trim data up to cut_stop
        _stops = scheduled_stops[vehicles[0]]
        if cut_stop in _stops:
            _idx = _stops.index(cut_stop) + 1
            _stops = _stops[:_idx]
        _y_idx = list(range(len(_stops)))

        # Generate a dynamic color palette
        palette = px.colors.qualitative.Plotly
        _vehicle_colors = {
            vid: palette[i % len(palette)]
            for i, vid in enumerate(vehicles)
        }

        _fig = go.Figure()

        # Scheduled (black solid)
        for _vid in vehicles:
            _fig.add_trace(go.Scatter(
                x=scheduled_times[_vid][:len(_stops)],
                y=_y_idx,
                mode='lines+markers',
                name=f"Sched {_vid}",
                line=dict(color='black', width=2),
                hovertemplate='Vehicle: %{name}<br>Time: %{x:.1f}s<br>Stop: %{text}',
                text=_stops
            ))

        # Actuals (solid vs dashed per vehicle)
        for _vid in vehicles:
            fixed_key  = f"{delay}s delay fixed block"
            moving_key = f"{delay}s delay moving block"
            for _scenario_name, _dash in [
                (fixed_key, "solid"), (moving_key, "dash")
            ]:
                _fig.add_trace(go.Scatter(
                    x=results_per_vehicle[_vid][_scenario_name][:len(_stops)],
                    y=_y_idx,
                    mode='lines+markers',
                    name=f"{_vid} ({_scenario_name})",
                    line=dict(color=_vehicle_colors[_vid], dash=_dash, width=1.5),
                    marker=dict(symbol='x'),
                    hovertemplate='Vehicle: %{name}<br>Time: %{x:.1f}s<br>Stop: %{text}',
                    text=_stops
                ))

        # Layout
        _fig.update_layout(
            title=f"Interactive Time–Space Diagram\n{', '.join(vehicles)} — {delay}s Fixed vs Moving (to {cut_stop})",
            xaxis=dict(title="Time (s)"),
            yaxis=dict(
                title="Stop",
                tickmode='array',
                tickvals=_y_idx,
                ticktext=_stops,
                autorange='reversed'
            ),
            legend=dict(orientation="v", x=1.02, y=1),
            margin=dict(l=80, r=200, t=100, b=80),
            hovermode='closest'
        )

        _fig.show()

    # Example call:
    #plot_interactive_time_space_dynamic(relevant_vehicles, delay, "Dornbusch_1", scheduled_stops,scheduled_times, results_per_vehicle1)

    #use cut stop to show only part of the diagram or use last station of you want to see it all
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Delay by station and vehicle""")
    return


@app.cell
def _(math, plt, results_per_vehicle1, scheduled_stops, scheduled_times):
    def plot_delay_compare(vehicles, delay):
        """
        Overlay arrival‐delay (no recovery) for multiple vehicles,
        comparing fixed vs. moving block for a given delay.
        """
        fixed_key  = f"{delay}s delay fixed block"
        moving_key = f"{delay}s delay moving block"
        stops      = scheduled_stops[vehicles[0]]
        cmap       = plt.get_cmap('tab10')

        fig, ax = plt.subplots(figsize=(10, 5))

        for idx, vid in enumerate(vehicles):
            sched = scheduled_times[vid]
            act_f = results_per_vehicle1[vid][fixed_key]
            act_m = results_per_vehicle1[vid][moving_key]

            # compute arrival delays in minutes, guard None → nan
            d_f = [
                (a - s)/60.0 if a is not None else math.nan
                for s, a in zip(sched, act_f)
            ]
            d_m = [
                (a - s)/60.0 if a is not None else math.nan
                for s, a in zip(sched, act_m)
            ]

            color = cmap(idx)
            # fixed‐block: solid
            ax.plot(stops, d_f,
                    marker='o', linestyle='-',
                    color=color,
                    label=f"{vid} conventional")
            # moving‐block: dashed
            ax.plot(stops, d_m,
                    marker='o', linestyle='--',
                    color=color,
                    label=f"{vid} CBTC")

        ax.set_xlabel("Station")
        ax.set_ylabel("Delay (min)")
        ax.set_title(f"Delay Comparison — {delay}s without and with CBTC", fontsize=16)
        ax.grid(True, linestyle=':')

        # Move legend outside to the right
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(), by_label.keys(),
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            fontsize = 14
        )


        plt.xticks(rotation=45, ha='right')
        #ax.xaxis.set_tick_params(labelbottom=False)

        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)  # make room for legend
        plt.show()
    return


@app.cell
def _(math, plt, results_per_vehicle1, scheduled_stops, scheduled_times):
    def plot_delay_compare_reversingline(
        vehicles, delay,
        reversing_pair=("Südbahnhof_0", "Südbahnhof_1"),
        reversing_label="Reversing",
        reversing_label_pos="top",  # 'top' or 'bottom'
        base_fontsize=12,
    ):
        """
        Overlay arrival‐delay (no recovery) for multiple vehicles,
        comparing fixed vs. moving block for a given delay.

        Adds a vertical dashed line and label for a reversing section
        between two stops (e.g. Südbahnhof_0 ↔ Südbahnhof_1).
        """

        fixed_key  = f"{delay}s delay fixed block"
        moving_key = f"{delay}s delay moving block"
        stops      = scheduled_stops[vehicles[0]]
        cmap       = plt.get_cmap('tab10')

        # Font setup (self-contained)
        plt.rcParams.update({
            "font.size": base_fontsize,
            "axes.titlesize": base_fontsize + 4,
            "axes.labelsize": base_fontsize + 2,
            "xtick.labelsize": base_fontsize,
            "ytick.labelsize": base_fontsize,
            "legend.fontsize": base_fontsize + 2,
        })

        fig, ax = plt.subplots(figsize=(10, 5))

        for idx, vid in enumerate(vehicles):
            sched = scheduled_times[vid]
            act_f = results_per_vehicle1[vid][fixed_key]
            act_m = results_per_vehicle1[vid][moving_key]

            # compute arrival delays in minutes, guard None → nan
            d_f = [
                (a - s)/60.0 if a is not None else math.nan
                for s, a in zip(sched, act_f)
            ]
            d_m = [
                (a - s)/60.0 if a is not None else math.nan
                for s, a in zip(sched, act_m)
            ]

            color = cmap(idx)
            # fixed‐block: solid
            ax.plot(stops, d_f,
                    marker='o', linestyle='-',
                    color=color,
                    label=f"{vid} conventional")
            # moving‐block: dashed
            ax.plot(stops, d_m,
                    marker='o', linestyle='--',
                    color=color,
                    label=f"{vid} CBTC")

        # --- Add vertical reversing line ---
        if reversing_pair is not None:
            a, b = reversing_pair
            if a in stops and b in stops:
                x0 = stops.index(a)
                x1 = stops.index(b)
                x_rev = (x0 + x1) / 2  # halfway between the two
                # Draw vertical dashed black line
                ymin, ymax = ax.get_ylim()
                ax.vlines(x=x_rev, ymin=ymin, ymax=ymax,
                          colors='black', linestyles='dashed', linewidth=1.8)

                # Place text label slightly below the top frame
                if reversing_label_pos.lower() == "top":
                    y_text = ymax - 0.05 * (ymax - ymin)  # shifted a bit down
                    va = 'top'
                else:
                    y_text = ymin + 0.05 * (ymax - ymin)
                    va = 'bottom'

                ax.text(x_rev, y_text, reversing_label,
                        color='black', fontsize=base_fontsize,
                        va=va, ha='center',
                        backgroundcolor='white')

                # keep limits unchanged
                ax.set_ylim(ymin, ymax)

        # --- Axes and labels ---
        ax.set_xlabel("Station")
        ax.set_ylabel("Delay (min)")
        ax.set_title(f"Delay Comparison — {delay}s without and with CBTC")
        ax.grid(True, linestyle=':')

        # --- Legend outside ---
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(), by_label.keys(),
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0
        )

        # Rotate x-labels for readability
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        plt.show()
    return (plot_delay_compare_reversingline,)


@app.cell
def _(delay, plot_delay_compare_reversingline, relevant_vehicles):
    plot_delay_compare_reversingline(relevant_vehicles, delay)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Einfaches Histogram Average Delay""")
    return


@app.cell
def _(plt):
    def plot_avg_delay_histogram(vehicles, delay,
                                 scheduled_times, results_per_vehicle,
                                 font_size=14, fmt="{:.1f}", xtick_pad=10):
        """
        vehicles            : list of vehicle IDs
        delay               : integer delay value (e.g. 600)
        scheduled_times     : dict {vid: [sched_time1, …]}       # times in seconds
        results_per_vehicle : dict {vid: {f"{delay}s delay fixed block": [...],
                                          f"{delay}s delay moving block": [...]} }
        font_size           : integer font size for all text
        fmt                 : format string for minute values (e.g. "{:.1f}")
        xtick_pad           : padding (in points) between x-tick labels and axis
        """
        # — bump all fonts —
        plt.rcParams.update({
            'font.size': font_size,
            'axes.labelsize': font_size,
            'xtick.labelsize': font_size,
            'ytick.labelsize': font_size,
            'legend.fontsize': font_size,
        })

        fixed_key  = f"{delay}s delay fixed block"
        moving_key = f"{delay}s delay moving block"

        avg_fixed_min  = []
        avg_moving_min = []
        for vid in vehicles:
            sched = scheduled_times[vid]
            act_f = results_per_vehicle[vid][fixed_key]
            act_m = results_per_vehicle[vid][moving_key]

            # compute delays (s), skip None, then avg and convert to minutes
            delays_f = [(a - s) for s, a in zip(sched, act_f) if a is not None]
            delays_m = [(a - s) for s, a in zip(sched, act_m) if a is not None]
            avg_fixed_min.append((sum(delays_f)/len(delays_f))/60 if delays_f else 0.0)
            avg_moving_min.append((sum(delays_m)/len(delays_m))/60 if delays_m else 0.0)

        n    = len(vehicles)
        pos  = range(n)
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))

        # draw bars and keep references
        bars_fixed  = ax.bar([p - width/2 for p in pos],
                             avg_fixed_min, width, label='conventional')
        bars_moving = ax.bar([p + width/2 for p in pos],
                             avg_moving_min, width, label='CBTC')

        ax.set_xticks(pos)
        ax.set_xticklabels(vehicles)
        ax.tick_params(axis='x', pad=xtick_pad)
        ax.set_xlabel('Vehicle')
        ax.set_ylabel('Average Delay (min)')
        ax.legend(loc='upper right')
        ax.grid(axis='y', linestyle=':')

        # annotate inside bars
        ann_font = font_size * 0.8
        for bar in list(bars_fixed) + list(bars_moving):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                h/2,
                fmt.format(h),
                ha='center', va='center',
                fontsize=ann_font,
                color='white'
            )

        plt.tight_layout()
        plt.show()
    return (plot_avg_delay_histogram,)


@app.cell
def _(
    delay,
    plot_avg_delay_histogram,
    relevant_vehicles,
    results_per_vehicle1,
    scheduled_times,
):
    plot_avg_delay_histogram(
        relevant_vehicles,
        delay,
        scheduled_times,
        results_per_vehicle1, # or results_per_vehicle_all, whichever you’re using
        font_size=22   
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Time to recovery on a vehicle basis""")
    return


@app.cell
def _(plt):
    def plot_recovery_time_histogram(
        vehicles,
        delay,
        scheduled_times,
        results_per_vehicle,
        require_permanent=False,   # False = first non-positive; True = must stay non-positive
        font_size=10,
        legend_outside=False,
        figsize=(8, 4),
    ):
        """
        Compute time to recovery per vehicle under fixed vs moving block.
        If require_permanent is True, recovery is when delay first becomes
        non-positive AND remains non-positive thereafter. Otherwise, it's
        the first time delay becomes non-positive. Times are capped at 60 min.
        """
        fixed_key  = f"{delay}s delay fixed block"
        moving_key = f"{delay}s delay moving block"
        cap_sec    = 60 * 60

        def compute_first_recovery(act, sched):
            first_sched = sched[0]
            for a, s in zip(act, sched):
                if a is not None and (a - s) <= 0:
                    return min(a - first_sched, cap_sec)
            return cap_sec

        def compute_permanent_recovery(act, sched):
            first_sched = sched[0]
            delays = [(a - s) if a is not None else None for s, a in zip(sched, act)]
            n = len(delays)
            for i in range(n):
                di = delays[i]
                if di is not None and di <= 0:
                    # Must be non-positive and non-None for all remaining points
                    ok = True
                    for dj in delays[i:]:
                        if dj is None or dj > 0:
                            ok = False
                            break
                    if ok:
                        return min(act[i] - first_sched, cap_sec)
            return cap_sec

        rec_fixed, rec_moving = [], []
        for vid in vehicles:
            sched = scheduled_times[vid]
            act_f = results_per_vehicle[vid][fixed_key]
            act_m = results_per_vehicle[vid][moving_key]

            if require_permanent:
                rf = compute_permanent_recovery(act_f, sched)
                rm = compute_permanent_recovery(act_m, sched)
            else:
                rf = compute_first_recovery(act_f, sched)
                rm = compute_first_recovery(act_m, sched)

            rec_fixed.append(rf / 60.0)
            rec_moving.append(rm / 60.0)

        # Plot
        pos = list(range(len(vehicles)))
        width = 0.35
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar([p - width/2 for p in pos], rec_fixed,  width, label='Fixed block')
        ax.bar([p + width/2 for p in pos], rec_moving, width, label='Moving block')

        ax.set_xticks(pos)
        ax.set_xticklabels(vehicles, fontsize=font_size)
        ax.set_xlabel('Vehicle', fontsize=font_size)
        ax.set_ylabel('Time to Recovery (min)', fontsize=font_size)
        ax.set_title(
            'Time to {}Recovery (cap 60 min) — {}s Delay'.format(
                'Permanent ' if require_permanent else '', delay
            ),
            fontsize=font_size + 2
        )
        ax.grid(axis='y', linestyle=':')

        if legend_outside:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=font_size)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
        else:
            ax.legend(loc='upper right', fontsize=font_size)
            plt.tight_layout()

        plt.show()
    return (plot_recovery_time_histogram,)


@app.cell
def _(
    delay,
    plot_recovery_time_histogram,
    relevant_vehicles,
    results_per_vehicle1,
    scheduled_times,
):
    plot_recovery_time_histogram(
        relevant_vehicles,
        delay,
        scheduled_times,
        results_per_vehicle1, 
        font_size=15,
        require_permanent=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Time to recovery overall""")
    return


@app.cell
def _(plt):
    def plot_global_recovery_histogram(
        vehicles,
        delays,
        scheduled_times,
        results_per_vehicle_all,
        tolerance_min=3,
        cap_min=60,
        *,
        font_size=16,
        legend_outside=False,
        figsize=(8, 4),
    ):
        """
        Static grouped bar chart of global recovery times (max over vehicles)
        for fixed vs moving block across multiple injected delays.
        'Recovery' means first time delay <= tolerance.
        Times are in minutes and capped at cap_min.
        """

        # Einheitliche Schriftgrößen für alle Elemente
        plt.rcParams.update({
            'font.size': font_size,
            'axes.titlesize': font_size,
            'axes.labelsize': font_size,
            'xtick.labelsize': font_size,
            'ytick.labelsize': font_size,
            'legend.fontsize': font_size,
        })

        tol_sec = tolerance_min * 60.0
        cap_sec = cap_min * 60.0

        global_fixed = []
        global_moving = []

        def recovery_time(act, sched):
            if not act or not sched:
                return cap_min
            if act[0] is not None and (act[0] - sched[0]) <= tol_sec:
                return 0.0
            first_sched = sched[0]
            for a, s in zip(act, sched):
                if a is not None and (a - s) <= tol_sec:
                    return min(a - first_sched, cap_sec) / 60.0
            return cap_min

        for d in delays:
            key_fixed  = f"{int(d)}s delay fixed block" if float(d).is_integer() else f"{d}s delay fixed block"
            key_moving = f"{int(d)}s delay moving block" if float(d).is_integer() else f"{d}s delay moving block"

            rec_f, rec_m = [], []
            for vid in vehicles:
                sched = scheduled_times[vid]
                act_f = results_per_vehicle_all[vid][key_fixed]
                act_m = results_per_vehicle_all[vid][key_moving]
                rec_f.append(recovery_time(act_f, sched))
                rec_m.append(recovery_time(act_m, sched))

            global_fixed.append(max(rec_f))
            global_moving.append(max(rec_m))

        # Plot
        x = list(range(len(delays)))
        width = 0.35
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar([xi - width/2 for xi in x], global_fixed, width, label='Fixed block')
        ax.bar([xi + width/2 for xi in x], global_moving, width, label='Moving block')

        ax.set_xticks(x)
        ax.set_xticklabels([f"{d}s" for d in delays])
        ax.set_xlabel('Injected Delay (s)')
        ax.set_ylabel('Time to Recovery (min)')
        ax.set_title(f'Global Recovery Time (tol {tolerance_min} min, cap {cap_min} min)')
        ax.grid(axis='y', linestyle=':')

        if legend_outside:
            legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout(rect=[0, 0, 0.85, 1])
        else:
            legend = ax.legend(loc='upper left')
            plt.tight_layout()

        plt.show()
    return (plot_global_recovery_histogram,)


@app.cell
def _(
    delays,
    plot_global_recovery_histogram,
    relevant_vehicles,
    results_per_vehicle_all,
    scheduled_times,
):
    plot_global_recovery_histogram(
         relevant_vehicles,
         delays,
         scheduled_times,
         results_per_vehicle_all,
         tolerance_min=0.5,
         cap_min=60
     )
    return


@app.cell
def _(delays, go, relevant_vehicles, results_per_vehicle_all, scheduled_times):
    def plot_global_recovery_interactive_tolerance(vehicles, delays,
                                         scheduled_times, results_per_vehicle_all,
                                         tolerance_min=1, cap_min=60):
        """
        Interactive grouped bar chart: global time until all vehicles are within tolerance,
        treating any initial delay ≤ tolerance as immediate recovery (0 min).

        vehicles                  : list of vehicle IDs
        delays                    : list of ints (e.g., [120,180,300,600])
        scheduled_times           : dict {vid: [sched_time1,...]} in seconds
        results_per_vehicle_all   : dict {vid: {scenario_key: [act_time1,...], ...}, ...}
        tolerance_min             : tolerance in minutes
        cap_min                   : cap in minutes if never reaches tolerance
        """
        tol_sec = tolerance_min * 60
        cap_sec = cap_min * 60
        global_fixed = []
        global_moving = []

        for d in delays:
            key_fixed  = f"{d}s delay fixed block"
            key_moving = f"{d}s delay moving block"

            rec_f = []
            rec_m = []
            for vid in vehicles:
                sched       = scheduled_times[vid]
                first_sched = sched[0]
                act_f       = results_per_vehicle_all[vid][key_fixed]
                act_m       = results_per_vehicle_all[vid][key_moving]

                def recovery_time(act):
                    for i, (a, s) in enumerate(zip(act, sched)):
                        if a is not None and (a - s) <= tol_sec:
                            # immediate if first stop is within tolerance
                            if i == 0:
                                return 0.0
                            # else time from first departure to recovery stop
                            return min((a - first_sched), cap_sec) / 60.0
                    return cap_min

                rec_f.append(recovery_time(act_f))
                rec_m.append(recovery_time(act_m))

            # global = max across vehicles
            global_fixed.append(max(rec_f))
            global_moving.append(max(rec_m))

        labels = [f"{d}s" for d in delays]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=global_fixed, name='Fixed block',
                             hovertemplate='%{x}<br>Fixed rec: %{y:.1f} min'))
        fig.add_trace(go.Bar(x=labels, y=global_moving, name='Moving block',
                             hovertemplate='%{x}<br>Moving rec: %{y:.1f} min'))

        fig.update_layout(
            barmode='group',
            title=f"Global Recovery Time (tol {tolerance_min} min, cap {cap_min} min)",
            xaxis_title="Injected Delay",
            yaxis_title="Time to Recovery (min)",
            legend_title="Block Type",
            hovermode='x'
        )
        fig.show()


    plot_global_recovery_interactive_tolerance(relevant_vehicles,delays,scheduled_times,results_per_vehicle_all,cap_min=60)
    return


if __name__ == "__main__":
    app.run()
