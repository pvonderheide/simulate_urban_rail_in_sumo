import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Run SUMO Simulation via TRACI and add a fixed Delay""")
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
    - Delay is added at runtime of the simulation.
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
def _(os):
    # load path names and keys from dotenv
    dir = os.getenv("scenario_dir")

    os.chdir(dir)
    os.getcwd()
    return


@app.cell
def _():
    #set if sumo runs with or without gui
    gui = True
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
    from dotenv import find_dotenv, load_dotenv, find_dotenv
    return find_dotenv, load_dotenv, mo, os, sys


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
    return


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
def _(os):
    # — parameters —
    trigger_step     = 620 # stop at Dornbusch 36 to 56
    delay_duration = 900     # seconds to hold at 0 m/s ---> 120 / 180 / 300 / 600


    # create a folder for the simulation results – named after added delay. 
    path = './'+ "fixedDelay"+ str(delay_duration)
    # create new single directory
    if not os.path.exists(path):
        os.makedirs(path)
    return (delay_duration,)


@app.cell
def _(delay_duration, prepare_for_traci, sumoConfig, traci):
    target_vehicle = "U2.1" # the ID of the vehicle you want to delay
    sumoBinary = prepare_for_traci()

    try: 
        #start the traci session
        traci.start([sumoBinary, "-c", sumoConfig, "--output-prefix", "fixedDelay"+str(delay_duration)+"/"])
        print("connected")

        step = 0

        while traci.simulation.getMinExpectedNumber() > 0:

            # --- once your target appears, issue the slowdown ---
            if target_vehicle in traci.vehicle.getIDList():
                # clamp its speed to 0 m/s for `delay_duration` seconds
                traci.vehicle.slowDown(
                    vehID=target_vehicle,
                    speed=0.0,
                    duration=delay_duration
                )
                print(f"→ slowed down {target_vehicle} for {delay_duration}s")
                # set to None so we don’t repeat it every step
                target_vehicle = None

            traci.simulationStep()
            step += 1

    #end traci session
    #always end traci session, even if error occurs, otherwise connection will stay active
    finally: 
        traci.close()
        print("disconnected")
    return


if __name__ == "__main__":
    app.run()
