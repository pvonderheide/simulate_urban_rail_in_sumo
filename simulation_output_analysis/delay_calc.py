""" created by Paula von der Heide, TUBS, 2025

If you use this code in your research or publications, please cite: 
"A SUMO-based study pf Urban Rail Operations on Frankfurts Corridor A", Paula von der Heide und Prof. Dr.-Ing. Lars Schnieder, 5.th International Railway Symposium Aachen, 2025
not for commercial use """


import xml.etree.ElementTree as ET
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def get_all_vehicle_ids_from_trips(schedule_file):
    """
    Parses the GTFS‐derived schedule XML and returns a list of all vehicle IDs.
    """
    tree = ET.parse(schedule_file)
    root = tree.getroot()
    # Find every <vehicle> element and grab its 'id' attribute
    vehicle_ids = [veh.get('id') for veh in root.findall('vehicle') if veh.get('id') is not None]
    return vehicle_ids

def parse_schedule_gtfs(schedule_file, vehicle_ids):
    """
    Returns a dict mapping each vehicle_id → list of (busStop, scheduled_time).
    """
    tree = ET.parse(schedule_file)
    root = tree.getroot()
    result = {vid: [] for vid in vehicle_ids}
    for veh in root.findall('vehicle'):
        vid = veh.get('id')
        if vid in result:
            for stop in veh.findall('stop'):
                bs = stop.get('busStop')
                scheduled = float(stop.get('until'))
                result[vid].append((bs, scheduled))

    # Warn if any requested ID not found
    missing = [vid for vid,stops in result.items() if not stops]

    if missing:
        print(f"[warning] no schedule stops found for: {', '.join(missing)}")
    return result

def load_stop_names_from_gtfs(stops_file):
    """
    Parses the GTFS‐derived stops additional file and returns
    a dict mapping each trainStop id → its human‐readable name.
    """
    tree = ET.parse(stops_file)
    root = tree.getroot()
    # find every <trainStop id="…" name="…"/> at any depth
    return {
        ts.get('id'): ts.get('name')
        for ts in root.findall('.//trainStop')
        if ts.get('id') and ts.get('name')
    }

def parse_actuals(actual_file, vehicle_ids):
    """
    Returns a dict mapping each vehicle_id → list of (busStop, actual_departure_time).
    """
    tree = ET.parse(actual_file)
    root = tree.getroot()
    result = {vid: [] for vid in vehicle_ids}
    # SUMO uses <stopinfo> tags
    for si in root.findall('stopinfo'):
        vid = si.get('id')
        if vid in result:
            ended = si.get('ended')
            if ended is not None:
                result[vid].append((si.get('busStop'), float(ended)))

    missing = [vid for vid,acts in result.items() if not acts]

    if missing:
        print(f"[warning] no actual departures found for: {', '.join(missing)}")
    return result

def combine(schedule_map, actual_map):
    """
    Returns a dict mapping each vehicle_id → list of dicts:
      { 'busStop', 'scheduled', 'actual':[...]}
    """
    combined_all = {}
    for vid, sched in schedule_map.items():
        actuals = actual_map.get(vid, [])
        # map each stop to list of actuals
        act_dict = {}
        for bs, t in actuals:
            act_dict.setdefault(bs, []).append(t)
        combined = []
        for bs, s in sched:
            combined.append({
                'busStop': bs,
                'scheduled': s,
                'actual': act_dict.get(bs, [])
            })
        combined_all[vid] = combined
    return combined_all

def load_stop_names_from_gtfs(stops_file):
    """
    Parses the GTFS‐derived stops additional file and returns
    a dict mapping each trainStop id → its human‐readable name.
    """
    tree = ET.parse(stops_file)
    root = tree.getroot()
    # find every <trainStop id="…" name="…"/> at any depth
    return {
        ts.get('id'): ts.get('name')
        for ts in root.findall('.//trainStop')
        if ts.get('id') and ts.get('name')
    }

def get_delays(combined_map, vehicle_ids=None):
    """
    Extracts per-stop arrival delay for one or more vehicles.

    Args:
      combined_map : dict { vehicle_id → list of entries }
                     each entry is {'busStop','scheduled','actual':[…]}
      vehicle_ids  : list of vehicle IDs to include; if None, uses all keys

    Returns:
      dict mapping each vehicle_id → list of dicts:
        {
          'stop_id' : str,
          'scheduled': float,
          'actual'   : float or None,
          'delay'    : float or None
        }
    """
    if vehicle_ids is None:
        vehicle_ids = combined_map.keys()

    delays = {}
    for vid in vehicle_ids:
        entries = combined_map.get(vid, [])
        vehicle_delays = []
        for e in entries:
            stop_id   = e['busStop']
            sched     = e['scheduled']
            actual    = (e['actual'][0] if e['actual'] else None)
            delay     = (actual - sched) if actual is not None else None
            vehicle_delays.append({
                'stop_id':   stop_id,
                'scheduled': sched,
                'actual':    actual,
                'delay':     delay
            })
        delays[vid] = vehicle_delays
    return delays

def extract_vehicle_ids_from_fcdout(fcd_output_file):
    vehicle_ids = set()
    tree = ET.parse(fcd_output_file)
    root = tree.getroot()

    for vehicle in root.iter('vehicle'):
        vehicle_ids.add(vehicle.get('id'))

    return sorted(list(vehicle_ids))

def create_schedule_from_flows(xml_file, selected_vehicle_ids):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    schedule = {}
    
    for flow in root.findall('flow'):
        flow_id = flow.get('id')
        begin = float(flow.get('begin'))
        period = float(flow.get('period'))
        end = float(flow.get('end'))
        stops = [(stop.get('busStop'), float(stop.get('until'))) for stop in flow.findall('stop')]

        num_vehicles = int((end - begin) / period)
        
        for vehicle_idx in range(num_vehicles):
            vehicle_id = f"{flow_id}.{vehicle_idx}"
            if vehicle_id in selected_vehicle_ids:
                vehicle_schedule = [(bus_stop, until + vehicle_idx * period) for bus_stop, until in stops]
                schedule[vehicle_id] = vehicle_schedule
    
    return schedule

def create_schedule_from_route_definition(xml_file, selected_vehicle_ids):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 1) Build a lookup from route-id → list of (stop_id, until)
    route_stops = {}
    for route in root.findall('route'):
        rid = route.get('id')
        stops = [
            (stop.get('busStop'), float(stop.get('until')))
            for stop in route.findall('stop')
            if stop.get('busStop') is not None
        ]
        route_stops[rid] = stops

    schedule = {}

    # 2) Now iterate flows, pulling stops via the route lookup
    for flow in root.findall('flow'):
        flow_id = flow.get('id')
        begin   = float(flow.get('begin'))
        period  = float(flow.get('period'))
        end     = float(flow.get('end'))
        rid     = flow.get('route')

        stops = route_stops.get(rid, [])
        num_vehicles = int((end - begin) / period)

        for idx in range(num_vehicles):
            vehicle_id = f"{flow_id}.{idx}"
            if vehicle_id in selected_vehicle_ids:
                offset = begin + idx * period
                vehicle_schedule = [
                    (stop_id, until + offset)
                    for stop_id, until in stops
                ]
                schedule[vehicle_id] = vehicle_schedule

    return schedule



def load_stop_names_from_stop_file(stops_file):
    """
    Parses the SUMO additional file of busStop definitions and returns
    a dict { stopID: stopName }.
    """
    tree = ET.parse(stops_file)
    root = tree.getroot()
    # every <busStop id="…" name="…"/>
    return {bs.get('id'): bs.get('name') for bs in root.findall('busStop')}

def compute_delay_stats_per_station(combined_map, stop_names_map=None, include_missing=False):
    """
    Same as compute_delay_stats, but with an extra 'stop_name' column,
    and preserving the original sort‐by‐mean ordering.
    """
    records = []
    for vid, entries in combined_map.items():
        for e in entries:
            sid   = e['busStop']
            sched = e['scheduled']
            if e['actual']:
                for act in e['actual']:
                    records.append({
                        'vehicle_id': vid,
                        'stop_id':    sid,
                        'scheduled':  sched,
                        'actual':     act,
                        'delay':      act - sched
                    })
            elif include_missing:
                records.append({
                    'vehicle_id': vid,
                    'stop_id':    sid,
                    'scheduled':  sched,
                    'actual':     np.nan,
                    'delay':      np.nan
                })

    df = pd.DataFrame(records, columns=[
        'vehicle_id','stop_id','scheduled','actual','delay'
    ])

    # Add stop_name column if map provided
    if stop_names_map is not None:
        df['stop_name'] = df['stop_id'].map(lambda x: stop_names_map.get(x, x))

    # Build stats exactly as before
    if df.empty:
        stats = pd.DataFrame(columns=[
            'stop_id','stop_name','count','mean','median','std','p90'
        ])
        return df, stats

    agg = df.groupby('stop_id', sort=False)['delay'].agg(
        count='count',
        mean='mean',
        median='median',
        std='std',
        p90=lambda x: x.quantile(0.9)
    ).reset_index()

    # Restore the original sort‐by‐mean descending
    agg = agg.sort_values('mean', ascending=False).reset_index(drop=True)

    # Inject stop_name into stats, preserving order
    if stop_names_map is not None:
        agg['stop_name'] = agg['stop_id'].map(lambda x: stop_names_map.get(x, x))

    # Reorder columns
    stats = agg[['stop_id', 'stop_name', 'count', 'mean', 'median', 'std', 'p90']]
    return df, stats

def percent_stops_under_delay(combined_map, max_delay_seconds=180, vehicle_ids=None):
    """
    Returns a dict mapping each vehicle_id → percentage of stops
    where delay < max_delay_seconds.

    Args:
      combined_map      : dict { vehicle_id → list of entries }
                          each entry is {'busStop','scheduled','actual':[...] }
      max_delay_seconds : threshold in seconds (default 180)
      vehicle_ids       : list of vehicle IDs to include; if None, uses all

    Returns:
      dict { vehicle_id → percentage (0–100) }
    """
    if vehicle_ids is None:
        vehicle_ids = combined_map.keys()

    percentages = {}
    for vid in vehicle_ids:
        print(f"Vehicle ID: {vid}")
        entries = combined_map.get(vid, [])
        total_stops = len(entries)
        if total_stops == 0:
            percentages[vid] = None
            continue

        # count how many stops were < max_delay_seconds late
        under_count = 0
        for e in entries:
            if e['actual']:
                delay = e['actual'][0] - e['scheduled']
                if delay < max_delay_seconds:
                    under_count += 1
            else: 
                print("No actuals recorded.")

        percentages[vid] = (under_count / total_stops) * 100
        print(f"total: {total_stops}")
        print(f"less than max seconds: {under_count}")

    return percentages

def compute_avg_delay_per_vehicle(combined_map):
    """
    Returns a dict mapping vehicle_id → average delay (in seconds)
    computed over all stops with an actual departure.
    """
    avg_delays = {}
    for vid, entries in combined_map.items():
        # collect all delays for this vehicle
        delays = [(e['actual'][0] - e['scheduled'])
                  for e in entries if e['actual']]
        #print(delays)
        if delays:
            avg_delays[vid] = sum(delays) / len(delays)
        else:
            avg_delays[vid] = None  # or 0.0 if you prefer
    return avg_delays

def find_missing_actuals(combined_map):
    """
    Returns a dict mapping each vehicle ID to a list of busStop IDs
    where no actual departure was recorded (i.e. actual == []).

    combined_map should be in the form:
      {
        'veh1': [
          {'busStop': 'stopA', 'scheduled': 3600, 'actual': [3650]},
          {'busStop': 'stopB', 'scheduled': 3660, 'actual': []},  # missing
          ...
        ],
        'veh2': [...],
        ...
      }
    """
    missing = {}
    for vid, entries in combined_map.items():
        # collect all stops with empty actual lists
        skips = [e['busStop'] for e in entries if not e['actual']]
        if skips:
            missing[vid] = skips
    return missing

def overall_percentage_under_delay(combined_map, max_delay_seconds=180):
    """
    Calculates the overall percentage of stops (only those with actual departures)
    where the actual departure was less than `max_delay_seconds` late.

    Args:
      combined_map        : dict { vehicle_id → list of entries }
                            each entry is {'busStop','scheduled','actual':[...] }
      max_delay_seconds   : threshold in seconds (default 180)

    Returns:
      float : percentage of stops under the delay threshold, or None if no actual stops
    """
    total_actual_stops = 0
    under_threshold = 0

    for entries in combined_map.values():
        for e in entries:
            if e['actual']:
                total_actual_stops += 1
                delay = e['actual'][0] - e['scheduled']
                if delay < max_delay_seconds:
                    under_threshold += 1

    if total_actual_stops == 0:
        return None
    return (under_threshold / total_actual_stops) * 100

def on_time_percentage_by_station(combined_map, stop_names_map, max_delay_seconds=180):
    """
    Computes the on‑time percentage per station, counting only observed stops
    (missing actuals are excluded from denominator).

    Returns a DataFrame with columns:
      stop_id, stop_name, observed_count, on_time_count, percent_on_time
    """
    stats = {}
    # Count only stops with actual departures
    for entries in combined_map.values():
        for e in entries:
            if not e['actual']:
                continue
            bs = e['busStop']
            delay = e['actual'][0] - e['scheduled']
            rec = stats.setdefault(bs, {'observed': 0, 'on_time': 0})
            rec['observed'] += 1
            if delay < max_delay_seconds:
                rec['on_time'] += 1

    rows = []
    for bs, c in stats.items():
        obs = c['observed']
        on_time = c['on_time']
        pct = (on_time / obs) * 100 if obs else 0
        rows.append({
            'stop_id': bs,
            'stop_name': stop_names_map.get(bs, bs),
            'observed_count': obs,
            'on_time_count': on_time,
            'percent_on_time': pct
        })

    df = pd.DataFrame(rows)
    return df.sort_values('percent_on_time', ascending=False).reset_index(drop=True)

def get_waiting_times(tripinfos_xml_file, vehicle_ids):
    """
    Returns a dictionary mapping vehicle IDs to their waiting times.

    Parameters:
        tripinfos_xml_file (str): Path to the tripinfos XML file.
        vehicle_ids (list or set): List of vehicle IDs to look up.

    Returns:
        dict: vehicle_id -> waiting time
    """
    tree = ET.parse(tripinfos_xml_file)
    root = tree.getroot()

    result = {}
    for tripinfo in root.findall('tripinfo'):
        vid = tripinfo.attrib.get('id')
        if vid in vehicle_ids:
            result[vid] = float(tripinfo.attrib.get('waitingTime', 0.0))
    return result


def get_average_waiting_time(tripinfos_xml_file):
    """
    Calculates the average waiting time of all vehicles in a SUMO tripinfos XML file.

    Parameters:
        tripinfos_xml_file (str): Path to the tripinfos XML file.

    Returns:
        float: Average waiting time in seconds, or None if no vehicles are found.
    """
    tree = ET.parse(tripinfos_xml_file)
    root = tree.getroot()

    waiting_times = []
    for tripinfo in root.findall('tripinfo'):
        waiting_time = float(tripinfo.attrib.get('waitingTime', 0))
        waiting_times.append(waiting_time)

    if not waiting_times:
        return None  # No vehicles found

    return sum(waiting_times) / len(waiting_times)
