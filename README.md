# simulate_urban_rail_in_sumo

Scripts are written in Python for marimo Notebook. 

requirements: 
- SUMO must be installed and PATH Variable of SUMO must be set (script uses SUMO python tooling). 
- download of GTFS schedule possible file: https://download.gtfs.de/germany/nv_free/latest.zip
- .env file with your paths pointing to your SUMO folders and sources

Scenario generation: 
- Create a SUMO scenario from Open Street Map and fit it for your railway simulation. Script downloads OSM via Overpass API (by relation ID or via explicit query) and runs necessary SUMO tools to convert it to SUMO Network. Possible adjustment of parametes (e.g. traffic types, vehicle types, platform length etc.)

Simulation scripts: 
- Add delay to an ongoing simulation
- Run a series of differnet delays and execute simualtion for each value automatically

Simulation output analysis: 
- Compare CBTC vs conventional with different delays. Evaluate average delay, delay across station, time space diagrams, Time to Recovery.
