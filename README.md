# 🌾 Digital Twin for Rural Sustainable Development — Phase 2
## Kalyandurg Village, Anantapur District, Andhra Pradesh

A multi-domain simulation platform integrating ML-driven prediction models for soil moisture, solar energy, crop yield, and population demand — with RAG-enhanced AI advisory and interactive 3D visualization.

---

## Quick Start

```bash
# 1. Install dependencies
pip install streamlit plotly scikit-learn pandas numpy joblib

# 2. (Optional) For RAG chatbot with Gemini
pip install langchain langchain-google-genai langchain-community faiss-cpu google-generativeai pypdf
export GOOGLE_API_KEY="Your key"

# 3. Generate datasets (or replace with real data - see below)
cd data && python generate_datasets.py && cd ..

# 4. Train models
python notebooks/train_models.py

# 5. Set up RAG knowledge base
python app/rag_system.py

# 6. Run the app
cd app && streamlit run streamlit_app.py
```

---

## Project Structure

```
digital-twin-v2/
├── data/
│   ├── raw/                          # Raw datasets
│   │   ├── nasa_power_kalyandurg_2024.csv    # Hourly weather (8784 records)
│   │   ├── icrisat_crop_data_anantapur.csv   # 20yr crop data (160 records)
│   │   ├── census_2011_kalyandurg_villages.csv # 20 villages
│   │   └── cgwb_groundwater_anantapur.csv    # 10yr groundwater (100 records)
│   ├── processed/
│   ├── generate_datasets.py          # Dataset generator (replace with real data)
│   └── fetch_nasa_power.py           # NASA POWER API fetcher
├── models/                           # Trained model files (.pkl)
├── notebooks/
│   └── train_models.py               # Full training pipeline
├── app/
│   ├── streamlit_app.py              # Main dashboard
│   ├── simulation_engine.py          # Core simulation loop
│   ├── rag_system.py                 # RAG chatbot system
│   └── visualizer_3d.html            # Three.js 3D visualization
├── rag/
│   └── documents/                    # PDFs for RAG knowledge base
└── README.md
```

---

## Getting REAL Data (Critical for Paper)

### 1. NASA POWER — Weather & Solar
```
URL: https://power.larc.nasa.gov/data-access-viewer/
Coords: 14.75°N, 77.11°E (Kalyandurg)
Params: T2M, T2M_MAX, T2M_MIN, RH2M, ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN, PRECTOTCORR, WS2M
Period: 2023-01-01 to 2024-12-31
Format: CSV
```

### 2. ICRISAT — Crop Production
```
URL: http://data.icrisat.org/dld/src/crops.html
Filter: Andhra Pradesh > Anantapur
Download: Area, Production, Yield for all crops
```

### 3. Census 2011 — Village Data
```
URL: https://data.gov.in/catalog/village-amenities-census-2011
Filter: Andhra Pradesh > Anantapur district
Also: https://censusindia.gov.in/census.website/data/population-finder
```

### 4. CGWB — Groundwater
```
URL: https://cgwb.gov.in/GW-data-access.html
Also: https://apwrims.ap.gov.in/mis/groundwater/levels
District: Anantapur
```

---

## Model Performance

| Model | Algorithm | R² Score | RMSE | Features |
|-------|-----------|----------|------|----------|
| Soil Moisture | GradientBoosting | 0.993 | 0.16 | 23 (lag, rolling, ET₀, cyclical) |
| Solar Power | MLP Neural Net | 0.998 | 60.9W | 14 (irradiance, clear-sky index, temporal) |
| Crop Yield | Random Forest | 0.964 | 122.5 kg/ha | 7 (rainfall, temp, crop type, area) |
| Pop. Demand | System Dynamics | N/A | N/A | Census baseline + growth projection |

---

## Architecture

```
Streamlit Dashboard ──── Plotly Charts + Three.js 3D
        │
   Simulation Engine ─── 30-day forward loop
        │
   ┌────┼────┐────────┐
   │    │    │        │
Moisture Solar Crop  Population
 Model  Model Model  Demand
 (GB)  (MLP) (RF)  (SysDyn)
   │    │    │        │
   └────┼────┘────────┘
        │
   Data Layer: NASA POWER + ICRISAT + Census + CGWB
        │
   RAG Knowledge Base ─── FAISS + Gemini API
```

---

## Team Task Assignment

| Member | Responsibilities |
|--------|-----------------|
| Person A | Data collection (NASA POWER API, ICRISAT download), Moisture + Crop models |
| Person B | Solar model, Population demand model, Scenario sliders |
| Person C | RAG system (FAISS + Gemini), Chatbot integration, Knowledge base curation |
| Person D | Streamlit dashboard, Three.js 3D viz, Plotly charts, Demo prep |

--- 

## Conference Paper Sections

1. **Introduction** — Rural sustainability challenges, digital twin concept
2. **Related Work** — Agricultural digital twins, RAG in agriculture, simulation engines
3. **System Architecture** — Multi-domain composable design
4. **Data Sources** — NASA POWER, ICRISAT, Census, CGWB
5. **Methodology** — Feature engineering, model selection, simulation logic
6. **Results** — Model metrics, simulation outputs, scenario comparisons
7. **RAG Evaluation** — Retrieval precision, response quality
8. **Discussion** — Scalability, limitations, real-world applicability
9. **Conclusion** — Contributions, future work
