# Simulate Urban Rail in SUMO
created by Paula von der Heide, TU Braunschweig, last updated November 2025
contact me at Paula.von-der-Heide@tu-braunscheig.de

If you use this code in your research or publications, please cite: "A SUMO-based study pf Urban Rail Operations on Frankfurts Corridor A", Paula von der Heide und Prof. Dr.-Ing. Lars Schnieder, 5.th International Railway Symposium Aachen, 2025. Not for commercial use.

This toolkit provides scripts used to generate an urban rail scenario with SUMO, add delay in a running simulation and analzye the effects of it afterwards. 

Requirements:
- SUMO must be installed and PATH Variable of SUMO must be set (script uses SUMO python tooling). 
- download of GTFS schedule possible file: https://download.gtfs.de/germany/nv_free/latest.zip
- We use a dotenv package setup to provide the path to the simulation files. Add an .env file with your paths pointing to your SUMO folders and sources
- Scripts are written in Python for marimo Notebook.
- All modules must be installed in a virtual environment. Marimo provides for an installation option upon first execution. 

Scenario generation:
- Create a SUMO scenario from Open Street Map and fit it for your railway simulation. Script downloads OSM via Overpass API (by relation ID or via explicit query) and runs necessary SUMO tools to convert it to SUMO Network. Possible adjustment of parametes (e.g. traffic types, vehicle types, platform length etc.)

Simulation scripts:
- Add delay to an ongoing simulation
- Run a series of differnet delays and execute simualtion for each value automatically

**Simulation output analysis: **
- Compare CBTC vs conventional with different delays. Evaluate average delay, delay across station, time space diagrams, Time to Recovery.
