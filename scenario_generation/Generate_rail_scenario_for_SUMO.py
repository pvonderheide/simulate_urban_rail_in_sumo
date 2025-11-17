import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Generate a SUMO Scenario with OSM data for railways""")
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
    - This script creates a SUMO simulation for one or more railway line(s) which are identified via their OSM relation ID
    - All needed files for SUMO are created and extended to fit a railway simulation
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## setup environment""")
    return


@app.cell
def _():
    #package management
    import marimo as mo
    import os
    import sys
    import overpy
    import requests
    import shutil
    import pandas
    import urllib
    from datetime import datetime
    import xml.etree.ElementTree as ET
    import sumolib
    from dotenv import find_dotenv, load_dotenv, find_dotenv
    return (
        ET,
        datetime,
        find_dotenv,
        load_dotenv,
        mo,
        os,
        requests,
        shutil,
        sumolib,
    )


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
    #directory in which SUMO files are to be created
    dir= dir = os.getenv("target_directory")

    # path to GUI view.xml file
    GUI_settings_file = os.getenv("GUI_settings_file_path")

    gtfszip = os.getenv("gtfs_zip")
    return GUI_settings_file, dir, gtfszip


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## define scenario""")
    return


@app.cell
def _(mo):
    scenario_name = mo.ui.text(placeholder="scenario", label="Set a scenario name, e.g. name of the city or area (no spaces, no special characters):")
    scenario_name
    return (scenario_name,)


@app.cell
def _(scenario_name):
    scenario = scenario_name.value
    if not scenario_name.value:
        scenario = "example_scenario"
    return (scenario,)


@app.cell
def _(scenario):
    prefix_input = scenario+"_in"
    prefix_output = scenario+"_out"
    print("All input files suffixed with: " + prefix_input)
    print("All output files suffixed with: " + prefix_output)
    return (prefix_output,)


@app.cell
def _(datetime, os):
    def create_folder(custom_name, dir):
        # Get the current date and time in the format YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Combine timestamp with the custom name
        folder_name = f"{timestamp}_{custom_name}"

        # Create the full path
        full_path = os.path.join(dir, folder_name)

        # Create the folder
        os.makedirs(full_path, exist_ok=True)

        print(f"Folder created: {full_path}")
        return full_path
    return (create_folder,)


@app.cell
def filenames(scenario):
    #filenames
    osm_file = scenario + ".osm"
    net_file = scenario + ".net.xml"
    stop_file = scenario + "_trainStops.add.xml"
    ptlines_file = scenario + "_ptlines.add.xml"
    #route_file = scenario + "_ptflows.rou.xml"
    rail_poi_file = scenario + "_railpoi.add.xml"
    conf_file = scenario +".sumocfg"
    veh_file="vtypes.xml"
    return conf_file, net_file, osm_file, rail_poi_file, stop_file, veh_file


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## topology""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    Simulation of an urban railway line(s)
    Download topology information and all available additional information from OSM. Identify via **Relation ID**:
    """
    )
    return


@app.cell
def _():
    osm_relations = [
        {"city": "Frankfurt", "line":"U2", "relationid": "66639,939199"},
        {"city": "Frankfurt", "line":"A", "relationid": "66638, 939218, 66639, 939199, 66615, 939219, 939116, 939117, 939150, 939151"},
        {"city": "Frankfurt", "line":"U4", "relationid": "66630,939237"},
        {"city": "Frankfurt", "line":"ALL", "relationid": "66638, 939218, 66639, 939199, 66615, 939219, 939116, 939117, 939150, 939151, 66630, 939237, 66631, 939239, 66633, 939254, 939255, 66632"},
        {"city": "Hamburg", "line":"U4", "relationid": "2872789, 2872790"},
        {"city":"Frankfurt", "line": "U3", "relationid": "66615,939219"}
    ]
    return (osm_relations,)


@app.cell
def _(mo, osm_relations):
    # Table display
    table = mo.ui.table(data =osm_relations)
    return (table,)


@app.cell
def _(table):
    table
    return


@app.cell
def _(table):
    # Flatten, split, deduplicate, strip whitespace
    relID = {
        id_str.strip()
        for row in table.value
        for id_str in row["relationid"].split(",")
    }

    # Convert to sorted list and join without quotes
    relID_str= ",".join(sorted(relID))
    return (relID_str,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Query data via **Overpass API**, download as xml and save to file -_->scenario.osm_""")
    return


@app.cell
def _():
    overpass_FrankfurtALL ="""

    [out:xml][timeout:600];

    /* 1) Passenger route relations (incl. U7 east) */
    relation(id:66638,939218,66639,939199,66615,
             939219,939116,939117,939150,939151,
             66630,939237,66631,939239,66633,939254,939255,66632,
             2872789,2872790)
             ->.routes;

    /* 2) Expand all members (ways + their nodes) */
    ( .routes; >>; ) -> .relmem_raw;

    /* 3) Trim to wide corridor (covers Hohemark ↔ Enkheim & workshops)
          lat: 50.08..50.27 ; lon: 8.52..8.85 */
    (
      node.relmem_raw(50.08,8.52,50.27,8.85);
      way.relmem_raw(50.08,8.52,50.27,8.85);
    ) -> .relmem;

    /* 4) Main-line seed (for proximity queries) */
    way.relmem
      ["railway"~"^(light_rail|subway)$"]
      ["operator"="VGF"]
      -> .main;

    /* 5) Operational tracks near the main line (strict VGF) */
    way(around.main:200)(50.08,8.52,50.27,8.85)
      ["railway"~"^(light_rail|subway)$"]
      ["service"~"^(crossover|siding|yard)$"]
      ["operator"="VGF"]
      -> .ops;

    /* 6) Enkheim pocket (back-of-station) */
    node.relmem_raw["name"~"Enkheim"] -> .enkheim_pts;
    way(around.enkheim_pts:350)
      ["railway"~"^(light_rail|subway)$"]
      ["operator"="VGF"]
      -> .enkheim_yard;

    /* 7) Bockenheimer Warte capture (allow missing operator locally) */
    node.relmem_raw["name"~"Bockenheimer Warte"] -> .bw_pts;
    way(around.bw_pts:300)
      ["railway"~"^(light_rail|subway)$"]
      ["operator"~"^vgf$",i] -> .bw_tracks_strict;
    way(around.bw_pts:300)
      ["railway"~"^(light_rail|subway)$"]
      ["operator"!~"."] -> .bw_tracks_unop;
    (.bw_tracks_strict; .bw_tracks_unop;) -> .bw_tracks;

    /* 8) Explicit inclusion (your missing element) */
    way(id:1238346189) -> .manual;

    /* 9) Stadtbahnzentralwerkstatt capture (east of Heerstraße) */
    (
      node(50.08,8.52,50.27,8.85)["name"~"Stadtbahnzentralwerkstatt"];
      way (50.08,8.52,50.27,8.85)["name"~"Stadtbahnzentralwerkstatt"];
      relation(50.08,8.52,50.27,8.85)["name"~"Stadtbahnzentralwerkstatt"];
    ) -> .szw_named;

    way(around.szw_named:600)
      ["railway"~"^(rail|light_rail|subway)$"]
      ["operator"~"^vgf$",i] -> .szw_tracks_strict;

    way(around.szw_named:600)
      ["railway"~"^(rail|light_rail|subway)$"]
      ["operator"!~"."] -> .szw_tracks_unop;

    (.szw_tracks_strict; .szw_tracks_unop;) -> .szw_tracks;

    /* 10) Heerstraße helper bubble (yard links near the junction) */
    node.relmem_raw["name"~"Heerstraße"] -> .heer_pts;
    way(around.heer_pts:500)
      ["railway"~"^(rail|light_rail|subway)$"]
      ["service"~"^(crossover|siding|yard)$"]
      (50.08,8.52,50.27,8.85)
      -> .heer_ops_all;
    way.heer_ops_all["operator"~"^vgf$",i] -> .heer_ops_vgf;
    way.heer_ops_all["operator"!~"."]      -> .heer_ops_unop;
    (.heer_ops_vgf; .heer_ops_unop;) -> .heer_ops;

    /* 11) Hohemark terminus capture (grab both tracks; allow missing operator) */
    (
      node.relmem_raw["name"~"Hohemark"];
      way .relmem["name"~"Hohemark"];
      relation.relmem_raw["name"~"Hohemark"];
    ) -> .hoh_named;

    way(around.hoh_named:450)
      ["railway"~"^(rail|light_rail|subway)$"]
      ["operator"~"^vgf$",i] -> .hoh_strict;

    way(around.hoh_named:450)
      ["railway"~"^(rail|light_rail|subway)$"]
      ["operator"!~"."] -> .hoh_unop;

    (.hoh_strict; .hoh_unop;) -> .hoh_tracks;

    /* 12) Assemble and output (verbose geometry like your working query) */
    (
      .relmem;
      .ops;
      .enkheim_yard;
      .bw_tracks;
      .manual;
      .szw_tracks;
      .heer_ops;
      .hoh_tracks;
    ) -> .all;

    (.all; >;);
    out geom;


    """
    return


@app.cell
def _(requests):
    def download_rel_from_osm(_overpass_string):
        _url = r"http://overpass-api.de/api/interpreter"
        _response = requests.post(_url, data=_overpass_string)

        _response.encoding = _response.apparent_encoding
        #print(_response.text)
        #print(_response.encoding)

        return _response
    return (download_rel_from_osm,)


@app.cell
def _(download_rel_from_osm):
    def create_osm_file(relID, osm_file, path):

        _overpass_string = f"""[out:xml][timeout:25]; rel(id:{relID}); (._;>>;); out;""" # use RelIDs
        #_overpass_string = overpass_FrankfurtAsouth
        #_overpass_string = overpass_FrankfurtALL

        _x = download_rel_from_osm(_overpass_string)
        print(_overpass_string)

        #save to file
        with open(path+'\\'+osm_file, 'w',encoding='utf-8') as _f:
            _f.write(_x.text)

        print("Created "+ osm_file)
    return (create_osm_file,)


@app.cell
def _(mo):
    trainstop_length = mo.ui.dropdown(
        options=["50","100", "200", "500"],
        value="100",  # default selected value
        label="Set default train stop length [m]",
        allow_select_none= False
    )
    trainstop_length
    return (trainstop_length,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    **convert osm file to SUMO-network**

    - extract topology --> _.net.xml_
    - extract stops --> _trainStops.add.xml_
    - extract public transport lines (which lines stops at which stop) --> _ptlines.add.xml_
    """
    )
    return


@app.cell
def _(net_file, os, osm_file, scenario, stop_file, trainstop_length):
    def run_netconvert():

        os.system("netconvert --osm-files " + osm_file
                              +" --output-file "+ net_file
                              +" --ptstop-output "+ stop_file
                              +" --ptline-output "+ scenario+"_ptlines.add.xml"
                              +" --speed-in-kmh true"
                              +" --railway.topology.repair.minimal"
                              +" --geometry.remove.keep-ptstops"
                              +" --railway.signal.guess.by-stops true"
                              +" --osm.stop-output.length.train " + trainstop_length.value
                              +" --osm.railsignals ALL"
                              +" --railway.topology.output " +scenario+"_railrepair.xml"
                              +" --geometry.remove true"#replace nodes which only define geometry points
                              +" --remove-edges.isolated true" # remove not connected edges
                 ) 

        print("Created "+net_file)
        print("Created "+stop_file)
        print("Created "+scenario+"_ptlines.add.xml")
        print("Created "+scenario+"_railrepair.xml")
    return (run_netconvert,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""add **railway additionals** (platforms, stop positions, signals, switches) _--> _railpoi.add.xml_""")
    return


@app.cell
def _(os, scenario):
    def add_railway_additionals():

        os.system('polyconvert --osm-files ' +scenario+'.osm'
                        +' --type-file "C:\\Program Files (x86)\\Eclipse\\Sumo\\data\\typemap\\osmPolyconvertRail.typ.xml"'
                        +' --output-file '+scenario+'_railpoi.add.xml')
        print('Created '+scenario+'_railpoi.add.xml')
    return (add_railway_additionals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""You can check and modify your network via NetEdit""")
    return


@app.cell
def _(ET):
    # netedit and pt2flows only work with bus stops. Open the file to correct it and save it again

    def convert_bus_stops_to_train_stops(input_file, output_file=None):
        # Parse the input XML file.
        tree = ET.parse(input_file)
        root = tree.getroot()

        for elem in root.iter():
            # Change tag name if it's a <busStop> tag
            if elem.tag == "busStop":
                elem.tag = "trainStop"

            # Change attribute name if it's a busStop attribute inside another tag
            if "busStop" in elem.attrib:
                elem.attrib["trainStop"] = elem.attrib.pop("busStop")

        # Write the modified XML tree to the output file with XML declaration.
        out_file = output_file if output_file else input_file
        tree.write(out_file, encoding="utf-8", xml_declaration=True)
    return


@app.cell
def _(ET, sumolib):
    # add distance along corridor for one route
    def add_distance_along_route(input_net: str,
                                 input_routes: str,
                                 output_net: str,
                                 route_id: str = None,
                                 include_internal: bool = False) -> None:
        """
        Reads a SUMO network and a routes file, selects one route (by ID or the first),
        computes cumulative 'distance' along its edges in order, and writes out a new
        .net.xml with those <edge distance="..."> annotations only for that route.

        Parameters
        ----------
        input_net : str
            Path to the existing SUMO network file (.net.xml).
        input_routes : str
            Path to the SUMO routes file (.rou.xml).
        output_net : str
            Path where the annotated network will be written.
        route_id : str, optional
            The ID of the route to process. If None, the first <route> is used.
        include_internal : bool, optional
            Whether to include internal edges (default False).
        """
        # 1. Load the net
        net = sumolib.net.readNet(input_net, withInternal=include_internal)

        # 2. Find the target route (first or by ID)
        target_edges = None
        for route in sumolib.xml.parse_fast(input_routes, 'route', ['id', 'edges']):
            if route_id is None or route.id == route_id:
                target_edges = route.edges.split()
                actual_route_id = route.id
                break

        if target_edges is None:
            raise ValueError(f"No route with id='{route_id}' found in {input_routes}")

        # 3. Compute cumulative distances along that route sequence
        total = 0.0
        distances = {}
        for eid in target_edges:
            edge = net.getEdge(eid)
            if edge is None:
                raise KeyError(f"Edge '{eid}' not found in network")
            distances[eid] = total
            total += edge.getLength()

        # 4. Parse original .net.xml and annotate only those edges
        tree = ET.parse(input_net)
        root = tree.getroot()
        for edge_elem in root.findall('edge'):
            eid = edge_elem.get('id')
            if eid in distances:
                edge_elem.set('distance', str(distances[eid]))

        # 5. Write out with XML declaration
        tree.write(output_net, encoding='utf-8', xml_declaration=True)
        print(f"Route '{actual_route_id}' distances applied; output written to {output_net}")
    return


@app.cell
def _(ET, sumolib):
    # add distance along all routes in the network - careful, there will be some mismatch if lines converge. Distance of last added route will have its distance value!
    def add_distance_for_all_routes(input_net: str,
                               input_routes: str,
                               output_net: str,
                               include_internal: bool = False) -> None:
        """
        Reads a SUMO network and a routes file, then for each route in turn
        computes cumulative distance along its edges and sets/overwrites the
        network's generic `distance` attribute on each <edge>. Edges only
        visited by earlier routes keep their values; edges visited multiple
        times get their value replaced by the last visit.

        Parameters
        ----------
        input_net : str
            Path to the existing SUMO network file (.net.xml).
        input_routes : str
            Path to the SUMO routes file (.rou.xml).
        output_net : str
            Path where the annotated network will be written.
        include_internal : bool, optional
            Whether to include internal edges when querying lengths.
        """
        # 1. Load the network for length lookups
        net = sumolib.net.readNet(input_net, withInternal=include_internal)

        # 2. Parse the .net.xml into an ElementTree and map edgeID -> element
        tree = ET.parse(input_net)
        root = tree.getroot()
        edge_elems = {e.get('id'): e for e in root.findall('edge')}

        # 3. Iterate every <route> in the .rou.xml
        for route in sumolib.xml.parse_fast(input_routes, 'route', ['edges']):
            total = 0.0
            for eid in route.edges.split():
                edge = net.getEdge(eid)
                if edge is None:
                    raise KeyError(f"Edge '{eid}' not found in network")
                elem = edge_elems.get(eid)
                if elem is None:
                    raise KeyError(f"No XML element for edge '{eid}'")
                # set or overwrite the generic distance attribute
                elem.set('distance', str(total))
                total += edge.getLength()

        # 4. Write the updated network back out
        tree.write(output_net, encoding='utf-8', xml_declaration=True)
    return (add_distance_for_all_routes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## traffic""")
    return


@app.cell
def _(mo):
    traffic_generation = mo.ui.dropdown(
        options=["GTFS traffic", "Flows"],
        value="Flows",  # default selected value
        label="Select how traffic is generated",
        allow_select_none= False
    )
    traffic_generation
    return (traffic_generation,)


@app.cell
def _(mo):
    # define interval of flows
    flow_interval_min = mo.ui.number(
            start=1,
            stop=15,
            step=1,
            value = 6,
            label="Select the interval of subway lines to be created (in minutes)")
    return (flow_interval_min,)


@app.cell
def _(flow_interval_min, traffic_generation):
    # Display conditionally based on the dropdown selection
    flow_interval_min  if (traffic_generation.value =="Flows") else None
    return


@app.cell
def _(net_file, os, scenario, stop_file):
    #convert lines to flows _--> _ptflows.rou.xml_
    def make_flows(flow_interval_min):
        flow_interval_s=str(flow_interval_min*60)

        os.system("ptlines2flows.py"
                              +" -n "+ net_file
                              +" -s "+ stop_file
                              +" -l "+ scenario+"_ptlines.add.xml"
                              +" -o "+ scenario+"_ptflows.rou.xml"
                              +" -p "+ flow_interval_s
                              +" --use-osm-routes")
        print("Created "+scenario+"_ptflows.rou.xml")
        print("Created flow of vehicles with interval "+flow_interval_s)

        _routefile=scenario+"_ptflows.rou.xml"

        return(_routefile)
    return (make_flows,)


@app.cell
def _(mo, os):
    gtfsfile_browser = mo.ui.file_browser(
        initial_path=os.getcwd(), multiple=False,
        label = "Select the GTFS File."
    )
    return (gtfsfile_browser,)


@app.cell
def _(gtfsfile_browser, traffic_generation):
    gtfsfile_browser if (traffic_generation.value =="GTFS traffic") else None
    return


@app.cell
def _(gtfszip, os, scenario):
    #read schedule from GTFS download and create vehicles and stations according to it
    def create_veh_from_gtfs():
        #Fahrplan von GTFS lesen und einspielen

        #gtfszip = str(gtfsfile_browser.path(index=0))  # returns a Path object
    
        os.system('python "C:\\Program Files (x86)\\Eclipse\\Sumo\\tools\\import\\gtfs\\gtfs2pt.py" --network '+scenario+'.net.xml --gtfs "' + gtfszip + '" --date 20250204 --osm-routes '+scenario+'_ptlines.add.xml --modes subway')

        _routefile = "gtfs_pt_vehicles.add.xml"
        _stop_file = "gtfs_pt_stops.add.xml"

        return (_routefile, _stop_file)
    return (create_veh_from_gtfs,)


@app.cell
def _(gtfsfile_browser):
    gtfsfile_browser.path(index=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## vehicles""")
    return


@app.cell
def _(mo):
    vehicle_model = mo.ui.dropdown(
        options=["SUMO standard subway", "VGF U5 Modeling approach 1", "VGF U5 Modeling approach 2"],
        value="SUMO standard subway",  # default selected value
        label="Select how vehicles are modeled",
        allow_select_none= False
    )
    vehicle_model
    return (vehicle_model,)


@app.cell
def _(mo):
    vehicle_length = mo.ui.dropdown(
        options=["25", "50", "75", "100"],
        value="50",  # default selected value
        label="Set train length [m]",
        allow_select_none= False
    )
    vehicle_length
    return (vehicle_length,)


@app.cell
def _(vehicle_length, vehicle_model):
    standard_vtype= '<vType id="subway" vClass="subway" length="'+vehicle_length.value+ '"/>'

    VGF_U5_type1= '<vType id="VGF_U5_type1" vClass="subway" carFollowModel="Rail" trainType="custom" maxPower="2350" maxTraction="150" resCoef_quadratic="0.00028" resCoef_linear="0.00003" resCoef_constant="1.670" length="'+vehicle_length.value+ '" maxSpeed="19.44"/>'

    VGF_U5_type2='<vType id="VGF_U5_type2" vClass="subway" carFollowModel="Rail" accel="1.3" decel="1.7" sigma="0" length="'+vehicle_length.value+ '" maxSpeed="19.44"/>'

    if vehicle_model.value =="VGF U5 Modeling approach 1":
        vtype_definition = VGF_U5_type1
        vtype_id= "VGF_U5_type1"
    elif vehicle_model.value =="VGF U5 Modeling approach 2":
        vtype_definition = VGF_U5_type2
        vtype_id= "VGF_U5_type2"
    else:
        vtype_definition = standard_vtype #Standard SUMO subway
        vtype_id= "subway"
    return vtype_definition, vtype_id


@app.function
def save_vtype_to_file(vtype_string: str, output_filename: str):
    content = f'''<?xml version="1.0" encoding="UTF-8"?>

<!-- generated by Generate_rail_scenario_for_SUMO.py -->

<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
    {vtype_string}
</additional>
'''
    with open(output_filename, 'w', encoding='utf-8') as file:
        file.write(content)


@app.cell
def _(ET):
    def replace_type_attribute(_rt_file, new_vtype, output_file =None):
        """
        Opens an XML file, replaces all 'type' attribute values with a new value, and saves the result.

        Parameters:
            _rt_file (str): Name of the input XML file with assignment of vehicles to routes.
            new_vtype (str): The new value to assign to all 'type' attributes.
            output_file (str, optional): Path to save the modified XML. If None, overwrites the original.
        """
        tree = ET.parse(_rt_file)
        root = tree.getroot()

        # Iterate over all elements that have a 'type' attribute
        for elem in root.iter():
            if 'type' in elem.attrib:
                elem.attrib['type'] = new_vtype

        # Save the modified tree
        save_path = output_file if output_file else _rt_file
        tree.write(save_path, encoding='utf-8', xml_declaration=True)
    return (replace_type_attribute,)


@app.cell
def _(ET):
    #routes file created from netconvert has some default vehicle definitions it it. If I create my own, definition is made twice and results in an error. So remove the standard vtypes from the routes file. 

    def remove_vtype_entries(input_path: str, output_path: str) -> None:
        """
        Remove all <vType> elements from the XML at input_path and save
        the cleaned XML to output_path.
        """
        tree = ET.parse(input_path)
        root = tree.getroot()
        # Iterate through all elements and remove any <vType> children
        for parent in root.iter():
            for child in list(parent):
                if child.tag == 'vType':
                    parent.remove(child)
        # Write the cleaned XML back to a file
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
    return (remove_vtype_entries,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## configuration of simulation""")
    return


@app.cell
def _(mo):
    delay_input = mo.ui.number(
            start=0,
            stop=50,
            step=1,
            value =30,
            label="Delay between simulation steps (ms):")
    delay_input
    return (delay_input,)


@app.cell
def _(delay_input):
    delay = delay_input.value
    return (delay,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""build **simulator conig file** -->.sumocfg""")
    return


@app.cell
def _(conf_file, delay, prefix_output):
    def write_config(net_file, route_files, add_files):

        _config = f"""<?xml version="1.0" encoding="UTF-8"?>

        <sumoConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

            <input>
                <net-file value="{net_file}" help="Load road network description from FILE"/>
                <route-files value="{route_files}" help="Load routes descriptions from FILE(s)"/>
                <additional-files value="{add_files}" help="Load further descriptions from FILE(s)"/>
            </input>

            <output>
                <tripinfo-output value="{prefix_output}_tripinfos.xml"/>

                <fcd-output value="{prefix_output}_fcdout.xml" type="FILE" help="Save the Floating Car Data"/> 
                <fcd-output.geo value="false" type="BOOL" help="Save the Floating Car Data using geo-coordinates (lon/lat)"/>
                <fcd-output.distance value="true" type="BOOL" help="Add kilometrage to the FCD output (linear referencing)"/>
                <fcd-output.acceleration value="true" type="BOOL" help="Add acceleration to the FCD output"/>

                <railsignal-block-output value="{prefix_output}_Railsignal-blocks.xml" type="FILE" help="Save railsignal-blocks into FILE"/>
                <railsignal-vehicle-output value="{prefix_output}_Railsignal-block-occ.xml" type="FILE" help="Occupancy information"/>

            	<stop-output value="{prefix_output}_stop-output.xml" type="FILE" help="Record stops and loading/unloading of passenger and containers for all vehicles into FILE"/>   

                <statistic-output value="{prefix_output}_statsout.xml"/>

            </output>

        	<processing>
        		<time-to-teleport value="-1" type="TIME" help="Specify how long a vehicle may wait until being teleported, defaults to 300, non-positive values disable teleporting"/>
        		<time-to-teleport.railsignal-deadlock value="-1" type="TIME" help="The waiting time after which vehicles in a rail-signal based deadlock are teleported"/>
        	</processing>



        	<configuration>
                <gui-settings-file value="view.xml"/>
            </configuration>

            <gui_only>
                <delay value="{delay}" help="Use FLOAT in ms as delay between simulation steps"/>
            </gui_only>

        </sumoConfiguration>"""

        with open(conf_file, "w",encoding="utf-8") as _f:
            _f.write(_config)
    return (write_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Run scenario generation""")
    return


@app.cell
def _(create_scenario):
    create_scenario()
    return


@app.cell
def _(
    GUI_settings_file,
    add_distance_for_all_routes,
    add_railway_additionals,
    conf_file,
    create_folder,
    create_osm_file,
    create_veh_from_gtfs,
    dir,
    flow_interval_min,
    make_flows,
    mo,
    net_file,
    os,
    osm_file,
    rail_poi_file,
    relID_str,
    remove_vtype_entries,
    replace_type_attribute,
    run_netconvert,
    scenario,
    shutil,
    stop_file,
    traffic_generation,
    veh_file,
    vtype_definition,
    vtype_id,
    write_config,
):
    def create_scenario(): 
        #change to working directory and validate it
        _path = create_folder(scenario, dir)
        os.chdir(_path)
        mo.output.append(f"Working directory:{os.getcwd()}")

        #-------------------TOPOLOGY ------------------------------
        #download osm file
        create_osm_file(relID_str, osm_file, _path)
        #mo.output.append(f"OSM-File for downloaded relations {relID}: {osm_file}")

        #convert to SUMO network
        run_netconvert()
        mo.output.append(f"""
            Netconvert done:
            Created {net_file}
            """)

        #add switches, signals, etc.
        add_railway_additionals()
        mo.output.append(f"Railway additionals done: Created {rail_poi_file}")

        #copy files needed to run (view.xml and latest.zip)
        shutil.copy2(GUI_settings_file,_path)

        #-------------------TRAFFIC ------------------------------
        if traffic_generation.value == "Flows": 
            #make flows from ptlines
            rt_file = make_flows(flow_interval_min.value)
            remove_vtype_entries(input_path=rt_file, output_path=rt_file)

            #prepare sumocfg settings for flow Traffic
            route_files_string = rt_file
            add_files_string = veh_file+ ','+ stop_file + "," + rail_poi_file

            mo.output.append(f"Generated traffic flows with interval {flow_interval_min.value} min.")
            mo.output.append(f"Stops : {stop_file}")
            mo.output.append(f"Routes file: {rt_file}")

        elif traffic_generation.value == "GTFS traffic":
            #create vehicles from gtfs
            mo.output.append("Start extracting GTFS data ...")
            (rt_file, stp_file) = create_veh_from_gtfs()

            #prepare sumocfg settings for GTFS Traffic
            route_files_string =""
            add_files_string = veh_file+ ","+ stp_file + "," + rt_file + "," + rail_poi_file
            mo.output.append(f"Generated traffic from GTFS.")
            mo.output.append(f"Stops : {stp_file}")
            mo.output.append(f"Routes file: {rt_file}")

        #vtypes
        save_vtype_to_file(vtype_definition, veh_file)
        replace_type_attribute(rt_file, vtype_id)        
        mo.output.append(f"Vehicle Type Declaration: {veh_file}")
        mo.output.append(f"Vehicle Type: {vtype_definition}")

        #-------------------NACHBEARBEITUNG ------------------------------  

        add_distance_for_all_routes(
            input_net= net_file,
            input_routes= rt_file,
            output_net=net_file,
            include_internal=False
        )
        print(f"Annotated network written to {net_file}")

        #add_distance_along_route(
        #    input_net=net_file,
        #    input_routes= rt_file,
        #    output_net="annotatedNet.net.xml",
        #    route_id=None,           # or e.g. route_id="route_42"
        #    include_internal=False
        #)

        #-------------------CONFIG ------------------------------
        #write SUMO config file
        write_config(net_file, route_files_string, add_files_string)
        mo.output.append(f"Created SUMO config file:{conf_file}")

        # go back to parent directory if creating another scenario / reuse this script
        parent = os.path.join(os.getcwd(), os.pardir)
        os.chdir(parent)
    return (create_scenario,)


if __name__ == "__main__":
    app.run()
