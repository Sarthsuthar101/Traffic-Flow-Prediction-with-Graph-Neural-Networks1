"""
Kaggle METR-LA Dataset Loader for TrafficGNN Pro
================================================
Handles loading, generating, and managing traffic datasets.
Supports: METR-LA (Kaggle), Gujarat (built-in), Custom uploads.
"""

import numpy as np
import pandas as pd
import networkx as nx
import os
import json
from datetime import datetime, timedelta
import random

# ─── Dataset Registry ──────────────────────────────────────────────────────────
DATASET_INFO = {
    "metr_la": {
        "name": "METR-LA (Kaggle)",
        "description": "Los Angeles Metropolitan Traffic — 207 highway sensors",
        "source": "Kaggle / DCRNN Paper",
        "url": "https://www.kaggle.com/datasets",
        "n_sensors": 207,
        "time_interval": "5 min",
        "features": ["speed", "flow", "occupancy"],
        "region": "Los Angeles, CA",
        "time_range": "Mar 2012 – Jun 2012",
        "icon": "🇺🇸",
    },
    "gujarat": {
        "name": "Gujarat Smart City",
        "description": "Gujarat State Traffic — 20 city sensors",
        "source": "Synthetic (Built-in)",
        "url": "",
        "n_sensors": 20,
        "time_interval": "1 hour",
        "features": ["flow", "speed", "occupancy"],
        "region": "Gujarat, India",
        "time_range": "Live Simulation",
        "icon": "🇮🇳",
    }
}

# ─── METR-LA Data Generator ───────────────────────────────────────────────────
def generate_metr_la_data(n_sensors=207, n_timesteps=288, seed=42):
    """
    Generate realistic METR-LA format traffic data.
    288 timesteps = 1 day at 5-min intervals.
    Speed in mph, Flow in vehicles/5min, Occupancy in %.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Time indices (5-min intervals over 24 hours)
    time_of_day = (np.arange(n_timesteps) % 288) / 12  # hours (0-24)
    
    data = {}
    for s in range(n_sensors):
        # Base speed varies by sensor (30-70 mph highway speeds)
        base_speed = 45 + np.random.uniform(-15, 25)
        
        # Morning rush (6:30-9:30 AM) — speed drops
        morning_dip = -np.exp(-0.5 * ((time_of_day - 8.0) / 1.2)**2) * (20 + np.random.uniform(0, 15))
        
        # Evening rush (4:00-7:30 PM) — speed drops more
        evening_dip = -np.exp(-0.5 * ((time_of_day - 17.5) / 1.5)**2) * (25 + np.random.uniform(0, 18))
        
        # Midday slight dip
        midday_dip = -np.exp(-0.5 * ((time_of_day - 12.5) / 2.0)**2) * 5
        
        # Night boost (faster speeds at night)
        night_boost = np.where((time_of_day < 5) | (time_of_day > 22), 8, 0)
        
        # Sensor noise
        noise = np.random.normal(0, 2.5, n_timesteps)
        
        speed = np.clip(base_speed + morning_dip + evening_dip + midday_dip + night_boost + noise, 5, 75)
        
        # Flow derived from speed (inverse relationship)
        # Higher flow = more vehicles but slower speeds during congestion
        max_flow = 40 + np.random.randint(0, 30)  # vehicles per 5 min
        flow_base = max_flow * (1 - (speed / 75) * 0.3)  # More flow when speed is lower
        morning_flow = np.exp(-0.5 * ((time_of_day - 8.0) / 1.5)**2) * 25
        evening_flow = np.exp(-0.5 * ((time_of_day - 17.5) / 1.8)**2) * 30
        flow_noise = np.random.normal(0, 3, n_timesteps)
        flow = np.clip(flow_base + morning_flow + evening_flow + flow_noise, 1, 80)
        
        # Occupancy (% of time detector is occupied)
        occupancy = np.clip(flow * 0.35 + np.random.normal(0, 1.5, n_timesteps), 0, 100)
        
        data[s] = {
            'speed': speed,    # mph
            'flow': flow,      # vehicles per 5 min
            'occupancy': occupancy  # percent
        }
    
    return data


def load_sensor_locations(data_dir=None):
    """Load METR-LA sensor locations from CSV."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    csv_path = os.path.join(data_dir, 'sensor_locations.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        return generate_sensor_locations()


def generate_sensor_locations(n_sensors=207, seed=42):
    """Generate realistic LA highway sensor locations."""
    np.random.seed(seed)
    random.seed(seed)
    
    freeways = [
        ('I-5', 34.05, -118.25, 0.003, -0.001, 45),
        ('I-10', 34.04, -118.35, 0.0005, 0.004, 40),
        ('I-110', 34.00, -118.28, 0.004, 0.0005, 25),
        ('I-405', 34.05, -118.47, 0.003, 0.001, 30),
        ('US-101', 34.08, -118.33, 0.002, -0.003, 30),
        ('I-710', 33.95, -118.21, 0.003, 0.0003, 15),
        ('I-605', 33.98, -118.09, 0.003, 0.0002, 12),
        ('SR-60', 34.01, -118.17, 0.0005, 0.004, 10),
    ]
    
    districts = ['LA-North', 'LA-Central', 'LA-West', 'Hollywood', 'Westside',
                 'Downtown', 'South-LA', 'SE-LA', 'East-LA', 'Glendale',
                 'Pasadena', 'San-Fernando', 'Long-Beach', 'Inglewood',
                 'Burbank', 'Torrance', 'Santa-Monica']
    
    sensors = []
    sid = 0
    for fname, base_lat, base_lon, dlat, dlon, count in freeways:
        for i in range(count):
            if sid >= n_sensors:
                break
            lat = base_lat + dlat * i + np.random.uniform(-0.008, 0.008)
            lon = base_lon + dlon * i + np.random.uniform(-0.008, 0.008)
            sensors.append({
                'sensor_id': sid,
                'sensor_name': f"{fname} Sensor-{i:03d}",
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'road_type': 'highway' if random.random() > 0.15 else 'arterial',
                'district': random.choice(districts)
            })
            sid += 1
    
    return pd.DataFrame(sensors)


def load_adjacency_edges(data_dir=None):
    """Load adjacency edges from CSV."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    csv_path = os.path.join(data_dir, 'adj_edges.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        return generate_adjacency_edges()


def generate_adjacency_edges(n_sensors=207, seed=42):
    """Generate adjacency edges based on sensor proximity."""
    np.random.seed(seed)
    locations = generate_sensor_locations(n_sensors, seed)
    
    edges = []
    lats = locations['latitude'].values
    lons = locations['longitude'].values
    
    for i in range(n_sensors):
        # Calculate distances to all other sensors
        dists = np.sqrt((lats - lats[i])**2 + (lons - lons[i])**2)
        dists[i] = np.inf  # exclude self
        
        # Connect to 3-5 nearest neighbors
        n_neighbors = random.randint(3, 5)
        nearest = np.argsort(dists)[:n_neighbors]
        
        for j in nearest:
            dist_km = round(dists[j] * 111, 2)  # approx degrees to km
            if dist_km > 0:
                weight = round(1.0 / (1.0 + dist_km), 4)
                edges.append({
                    'from_sensor': i,
                    'to_sensor': int(j),
                    'distance_km': dist_km,
                    'weight': weight
                })
    
    df = pd.DataFrame(edges)
    # Remove duplicate edges (keep unique pairs)
    df = df.drop_duplicates(subset=['from_sensor', 'to_sensor'])
    return df


def build_graph_from_edges(edge_df, n_sensors=207):
    """Build NetworkX graph from edge DataFrame."""
    G = nx.Graph()
    G.add_nodes_from(range(n_sensors))
    
    for _, row in edge_df.iterrows():
        G.add_edge(
            int(row['from_sensor']),
            int(row['to_sensor']),
            distance=row['distance_km'],
            weight=row['weight']
        )
    
    return G


def build_metr_la_graph(n_sensors=207, seed=42):
    """Build complete METR-LA graph with node data."""
    locations = load_sensor_locations()
    edges = load_adjacency_edges()
    G = build_graph_from_edges(edges, n_sensors)
    
    pos = nx.spring_layout(G, seed=seed, k=2)
    
    node_data = {}
    for idx, row in locations.iterrows():
        sid = row['sensor_id']
        if sid >= n_sensors:
            break
        x, y = pos.get(sid, (0, 0))
        node_data[sid] = {
            'name': row['sensor_name'],
            'x': float(x),
            'y': float(y),
            'lat': row['latitude'],
            'lon': row['longitude'],
            'type': row['road_type'],
            'capacity': random.randint(1200, 3500),
            'city': row.get('district', 'LA'),
        }
    
    edge_data = {}
    for _, row in edges.iterrows():
        u, v = int(row['from_sensor']), int(row['to_sensor'])
        if G.has_edge(u, v):
            edge_data[(u, v)] = {
                'distance': row['distance_km'],
                'travel_time': round(row['distance_km'] / 60 * 60, 1),
                'road_type': 'highway'
            }
    
    return G, node_data, edge_data


def get_metr_la_traffic(traffic_data, sensors, t=None):
    """Get current traffic state for METR-LA dataset."""
    if t is None:
        t = datetime.now().hour + datetime.now().minute / 60
    
    # Convert hour to 5-min index (288 per day)
    idx_5min = min(int(t * 12), 287)
    
    results = []
    for s in range(len(sensors)):
        if s not in traffic_data:
            continue
        speed_mph = traffic_data[s]['speed'][idx_5min]
        speed_kmh = speed_mph * 1.60934  # Convert to km/h
        flow_5min = traffic_data[s]['flow'][idx_5min]
        flow_hr = flow_5min * 12  # Convert to veh/h
        occ = traffic_data[s]['occupancy'][idx_5min]
        
        capacity = sensors[s]['capacity']
        ratio = flow_hr / capacity
        
        if ratio > 0.9:
            status = 'CRITICAL'
        elif ratio > 0.7:
            status = 'CONGESTED'
        elif ratio > 0.5:
            status = 'MODERATE'
        else:
            status = 'FREE'
        
        results.append({
            'sensor_id': s,
            'name': sensors[s]['name'],
            'flow': int(flow_hr),
            'speed': round(speed_kmh, 1),
            'occupancy': round(occ, 1),
            'capacity': capacity,
            'congestion_ratio': round(ratio, 3),
            'status': status,
            'lat': sensors[s]['lat'],
            'lon': sensors[s]['lon'],
            'speed_mph': round(speed_mph, 1),
            'flow_5min': round(flow_5min, 1),
        })
    
    return pd.DataFrame(results)


def load_uploaded_csv(uploaded_file):
    """Load a user-uploaded traffic CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return None, str(e)


def get_dataset_summary(dataset_key):
    """Return formatted dataset info."""
    info = DATASET_INFO.get(dataset_key, {})
    return info


def generate_metr_la_week(n_sensors=207, seed=42):
    """Generate 1 week of METR-LA data (7 days × 288 = 2016 timesteps)."""
    np.random.seed(seed)
    random.seed(seed)
    
    n_timesteps = 288 * 7  # 1 week
    time_of_day = (np.arange(n_timesteps) % 288) / 12  # hours
    day_of_week = (np.arange(n_timesteps) // 288) % 7  # 0=Mon, 6=Sun
    is_weekend = (day_of_week >= 5).astype(float)
    
    data = {}
    for s in range(n_sensors):
        base_speed = 45 + np.random.uniform(-15, 25)
        
        morning_dip = -np.exp(-0.5 * ((time_of_day - 8.0) / 1.2)**2) * (20 + np.random.uniform(0, 15))
        evening_dip = -np.exp(-0.5 * ((time_of_day - 17.5) / 1.5)**2) * (25 + np.random.uniform(0, 18))
        
        # Weekends have less congestion
        weekend_relief = is_weekend * 10  # speed is faster on weekends
        
        noise = np.random.normal(0, 2.5, n_timesteps)
        speed = np.clip(base_speed + morning_dip + evening_dip + weekend_relief + noise, 5, 75)
        
        max_flow = 40 + np.random.randint(0, 30)
        flow_base = max_flow * (1 - (speed / 75) * 0.3)
        morning_flow = np.exp(-0.5 * ((time_of_day - 8.0) / 1.5)**2) * 25 * (1 - is_weekend * 0.4)
        evening_flow = np.exp(-0.5 * ((time_of_day - 17.5) / 1.8)**2) * 30 * (1 - is_weekend * 0.3)
        flow = np.clip(flow_base + morning_flow + evening_flow + np.random.normal(0, 3, n_timesteps), 1, 80)
        
        occupancy = np.clip(flow * 0.35 + np.random.normal(0, 1.5, n_timesteps), 0, 100)
        
        data[s] = {'speed': speed, 'flow': flow, 'occupancy': occupancy}
    
    return data
