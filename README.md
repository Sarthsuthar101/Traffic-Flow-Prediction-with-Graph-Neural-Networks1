# 🚦 TrafficGNN Pro — Smart City Traffic Intelligence

A full-stack, real-time **Traffic Flow Prediction** system using **Graph Neural Networks (GNN)**, with support for **Kaggle METR-LA dataset** (207 LA highway sensors) and **Gujarat Smart City** (20 city sensors), built with Streamlit.

---

## 📂 Kaggle Dataset Integration

This project supports the **METR-LA** traffic dataset from Kaggle — one of the most widely used benchmarks for traffic forecasting with GNNs.

| Property | Value |
|----------|-------|
| **Dataset** | METR-LA (Los Angeles Metropolitan Traffic) |
| **Source** | [Kaggle](https://www.kaggle.com/datasets) / DCRNN Paper (Li et al., ICLR 2018) |
| **Sensors** | 207 loop detectors on LA highway system |
| **Time Range** | March 1, 2012 – June 30, 2012 |
| **Interval** | 5-minute recordings |
| **Features** | Speed (mph), Flow (veh/5min), Occupancy (%) |
| **Timesteps** | 34,272 |

### How to Use Real Kaggle Data
1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Search for **"METR-LA"** or **"PEMS-BAY"**
3. Download the dataset (CSV format)
4. Upload via the **📂 Kaggle Dataset** page in the app
5. The app automatically adapts to your data

> **Note:** The app includes a pre-generated METR-LA format sample dataset that works out-of-the-box without needing a Kaggle account.

---

## 🏙️ Gujarat Cities Covered (20 sensors)

| # | City | Road/Location |
|---|------|--------------|
| 1 | Ahmedabad | SG Highway |
| 2 | Surat | Ring Road |
| 3 | Vadodara | RC Dutt Road |
| 4 | Rajkot | Kalawad Road |
| 5 | Gandhinagar | Sector-21 |
| 6 | Bhavnagar | Waghawadi Road |
| 7 | Jamnagar | NH-27 |
| 8 | Junagadh | Dhal Road |
| 9 | Anand | Karamsad Road |
| 10 | Mehsana | NH-48 |
| 11 | Nadiad | Station Road |
| 12 | Morbi | GIDC Road |
| 13 | Surendranagar | NH-8A |
| 14 | Botad | Bhavnagar Road |
| 15 | Amreli | Dhari Road |
| 16 | Porbandar | Marine Drive |
| 17 | Dwarka | NH-51 |
| 18 | Valsad | Tithal Road |
| 19 | Navsari | Surat Road |
| 20 | Bharuch | GIDC Road |

---

## 📦 Project Structure

```
traffic_gnn_app/
├── app.py                  ← Main Streamlit application
├── data_loader.py          ← Kaggle dataset loader module
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
└── data/
    ├── sensor_locations.csv ← METR-LA sensor coordinates (207 sensors)
    ├── adj_edges.csv        ← Sensor adjacency edge list
    └── README.md            ← Dataset documentation
```

---

## 🚀 Deploy on Streamlit Cloud (Free)

```bash
git init
git add .
git commit -m "TrafficGNN Pro — initial commit"
git remote add origin https://github.com/YOUR_USERNAME/traffic-gnn-pro.git
git push -u origin main
```

Then go to https://streamlit.io/cloud → New app → connect repo → set `app.py` → Deploy.

---

## 🔧 Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Features

- 🏠 **Live Dashboard** — Real-time KPIs for all sensors
- 🕸️ **Graph Network** — GNN road network visualization
- 📊 **GNN Prediction** — Multi-step flow forecasting per sensor
- 📂 **Kaggle Dataset** — METR-LA dataset explorer, upload, benchmarks
- 🚗 **Live Car View** — Animated vehicle simulation
- 🛰️ **Real-Time Auto Map** — Leaflet.js map with live cars
- 📹 **Live CCTV Camera** — Webcam/video motion detection
- 🌐 **Live Web View** — External site embed with overlay
- 🛤️ **Car History** — Vehicle path replay with timestamps
- 🏙️ **Smart City Apps** — Route optimization, congestion alerts, smart parking
- 📈 **Analytics** — OD matrix, sensor correlation, speed-flow diagrams
- 📥 **Download Center** — CSV/JSON/ZIP exports

---

## 📂 Supported Datasets

| Dataset | Sensors | Region | Status |
|---------|---------|--------|--------|
| METR-LA | 207 | Los Angeles, CA | ✅ Integrated |
| PEMS-BAY | 325 | San Francisco Bay | 📋 Compatible |
| PEMS-08 | 170 | San Bernardino | 📋 Compatible |
| Gujarat Smart City | 20 | Gujarat, India | ✅ Built-in |
| Custom Upload | Any | Any | ✅ Supported |

---

*TrafficGNN Pro — Built for Smart Cities with Kaggle Dataset Support*
