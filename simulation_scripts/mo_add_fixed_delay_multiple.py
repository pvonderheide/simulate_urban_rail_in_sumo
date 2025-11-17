import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Run SUMO Simulation via TRACI and vary the added delay""")
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
        """
    - This script runs a SUMO simulation (select via giving the folder of input files) via TRACI.
    - Provide a list of delay values that shall be added
    - Determine which vehicle shall be delayed
    - This script runs a simulation run for each of the delay values and saves results in a different folder ("fixedDelayXX")
    - This script runs a basic analysis and compares results of simulation runs
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""select **scenario** (give folder name in which SUMO files are saved)""")
    return


@app.cell
def _():
    sumoConfig = "FrankfurtAsouth.sumocfg"
    return (sumoConfig,)


@app.cell
def _(dir, os):
    os.chdir(dir)
    os.getcwd()
    return


@app.cell
def _():
    #set if sumo runs with or without gui
    gui = False
    return (gui,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## setup workspace""")
    return


@app.cell
def _():
    import marimo as mo
    import os
    import sys
    import urllib
    import random
    import pandas as pd
    from dotenv import find_dotenv, load_dotenv, find_dotenv
    return find_dotenv, load_dotenv, mo, os, pd, sys


@app.cell
def _(os, sys):
    # we need to import SUMO modules
    from sumolib import checkBinary  # noqa
    import traci  # noqa
    import traci.constants as tc

    # we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    return checkBinary, traci


@app.cell
def _(find_dotenv, load_dotenv, os):
    # find my dotenv file
    notebook_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(notebook_dir)

    dotenv_path = find_dotenv(usecwd=True)  # <- important
    load_dotenv(dotenv_path, override=True) # override true otherwise once loaded variables will never update

    dotenv_path

    dir = os.getenv("scenario_dir")
    path_to_sources=os.getenv("SUMO_src")
    return dir, path_to_sources


@app.cell
def _(path_to_sources, sys):
    sys.path.append(path_to_sources)
    import simulation_output_analysis.delay_visualization as vd
    import simulation_output_analysis.delay_calc as cd
    return (cd,)


@app.cell
def _():
    routes_file = "FrankfurtAsouth_flows_stops.rou.xml"
    stop_file ="FrankfurtAsouth_trainStops.add.xml"
    return routes_file, stop_file


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## connect with SUMO simulator""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""start SUMO via TRACI""")
    return


@app.cell
def _(checkBinary, gui, traci):
    def prepare_for_traci():
        """prepare connecting to traci"""

        #makes sure previous tracis are closed (in case of an error)
        if not (traci.isLoaded): 
            print(traci.isLoaded)
            print("was connected.")
            traci.close()
            print("now closed.")
        else:
            print("Nothing connected")

        #get sumoBinary to start Traci
        if gui:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')

        return sumoBinary
    return (prepare_for_traci,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## run simulation and add fixed delay""")
    return


@app.cell
def _(os, prepare_for_traci, sumoConfig, traci):
    def run_simulation(delay_duration, target_veh_ID):

        # create a folder for save the simulation results – named after added delay. 
        path = './'+ "fixedDelay"+ str(delay_duration)
        # create new single directory
        if not os.path.exists(path):
            os.makedirs(path)   

        try:
            sumoBinary = prepare_for_traci()

            # Start the traci session
            traci.start([sumoBinary, "-c", sumoConfig, "--output-prefix", "fixedDelay"+str(delay_duration)+"/"])
            print(f"Connected — running delay_duration = {delay_duration}s")

            step = 0
            _target_vehicle = target_veh_ID  # Reset for each run

            while traci.simulation.getMinExpectedNumber() > 0:

                # Issue the slowdown once the target appears
                if _target_vehicle in traci.vehicle.getIDList():
                    traci.vehicle.slowDown(
                        vehID=_target_vehicle,
                        speed=0.0,
                        duration=delay_duration
                    )
                    print(f"→ Slowed down {_target_vehicle} for {delay_duration}s")
                    _target_vehicle = None  # Prevent repeated slowdown

                traci.simulationStep()
                step += 1

        finally:
            traci.close()
            print("Disconnected\n")
    return (run_simulation,)


@app.cell
def _(cd, relevant_vehicles, routes_file, stop_file):
    def analyze_simulation_results(delay_duration):
        stop_output_file = "fixedDelay"+str(delay_duration)+"/"+"FrankfurtAsouth_out_stop-output.xml"
        fcd_output_file  = "fixedDelay"+str(delay_duration)+"/"+"FrankfurtAsouth_out_fcdout.xml"
        tripinfos_output_file = "fixedDelay"+str(delay_duration)+"/"+"FrankfurtAsouth_out_tripinfos.xml"

        stop_names_map = cd.load_stop_names_from_stop_file(stop_file)

        # Prepare maps for relevant vehicles
        sched_map_vids = cd.create_schedule_from_route_definition(routes_file, relevant_vehicles)
        actual_map_vids = cd.parse_actuals(stop_output_file, relevant_vehicles)
        combined_map_vids = cd.combine(sched_map_vids, actual_map_vids)

        # All vehicles
        all_vehicles = cd.extract_vehicle_ids_from_fcdout(fcd_output_file)
        sched_map_all = cd.create_schedule_from_route_definition(routes_file, all_vehicles)
        actual_map_all = cd.parse_actuals(stop_output_file, all_vehicles)
        combined_map_all = cd.combine(sched_map_all, actual_map_all)

        # Compute punctuality
        punctuality_map = cd.percent_stops_under_delay(combined_map_vids, max_delay_seconds=180)
        punctuality_all = cd.overall_percentage_under_delay(combined_map_all, max_delay_seconds=180)

        # Compute average delay
        avg_delays = cd.compute_avg_delay_per_vehicle(combined_map_vids)

        # Compute waiting times
        waiting_time_target_veh = cd.get_waiting_times(tripinfos_output_file, relevant_vehicles)
        waiting_time_all = cd.get_average_waiting_time(tripinfos_output_file)

        # Collect results
        result = {
            "delay_duration": delay_duration,
            "U2.1_punctuality": punctuality_map.get("U2.1", 0.0),
            "U2.1_avg_delay": avg_delays.get("U2.1", None),
            "U2.1_waiting_time": waiting_time_target_veh.get("U2.1", None),
            "U3.1_punctuality": punctuality_map.get("U3.1", 0.0),
            "U3.1_avg_delay": avg_delays.get("U3.1", None),
            "U3.1_waiting_time": waiting_time_target_veh.get("U3.1", None),
            "U1.2_punctuality": punctuality_map.get("U1.2", 0.0),
            "U1.2_avg_delay": avg_delays.get("U1.2", None),
            "U1.2_waiting_time": waiting_time_target_veh.get("U1.2", None),
            "All_punctuality": punctuality_all,
            "All waiting_time": waiting_time_all
        }

        return result
    return (analyze_simulation_results,)


@app.cell
def _():
    target_vehicle_id = "U2.1"  # the ID of the vehicle you want to delay
    delay_durations = [120, 180, 300,420, 600, 900, 1200, 1800]  # different delay durations

    # which vehicles to analyze
    delayed_vehicle = "U2.1"
    successor_1 = "U3.1"
    successor_2 = "U1.2"
    relevant_vehicles = [delayed_vehicle, successor_1, successor_2]
    return delay_durations, relevant_vehicles, target_vehicle_id


@app.cell
def _(
    analyze_simulation_results,
    delay_durations,
    pd,
    run_simulation,
    target_vehicle_id,
):
    results = []

    # Run the simulation for each delay duration
    for delay in delay_durations:
        run_simulation(delay, target_vehicle_id)
        result = analyze_simulation_results(delay)
        results.append(result)

    # Create and display a table
    df = pd.DataFrame(results)
    df_flipped = df.set_index("delay_duration").T

    # Save to XLSX
    with pd.ExcelWriter("simulation_results.xlsx") as writer:
        df_flipped.to_excel(writer, sheet_name="Results")

    # Optional: also print to terminal if needed
    print(df_flipped.to_string())
    return


if __name__ == "__main__":
    app.run()
