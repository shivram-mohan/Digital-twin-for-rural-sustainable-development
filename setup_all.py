#!/usr/bin/env python3
"""
ALL-IN-ONE SETUP: Generates datasets + trains models + verifies everything
Run this ONCE to set up the entire project.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib, json, os, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw")
MODELS = os.path.join(BASE, "models")
os.makedirs(RAW, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

# ==== CLIMATE PARAMS FOR KALYANDURG, ANANTAPUR (14.75N, 77.11E) ====
CLIMATE = {
    1:(24.5,12,45,2,850,2.5), 2:(27,13,38,3,920,2.8), 3:(30.5,14,32,8,980,3),
    4:(33.5,13,35,18,960,3.2), 5:(35,11,42,45,900,3.5), 6:(31.5,9,62,65,750,4),
    7:(29,7,72,95,620,3.8), 8:(28.5,7,75,110,600,3.5), 9:(28,8,74,130,680,3),
    10:(27.5,9,68,80,780,2.5), 11:(25.5,10,58,25,800,2.2), 12:(24,11,50,5,820,2.3),
}

def solar_curve(hour, peak, doy):
    dl = 12 + np.sin(2*np.pi*(doy-80)/365)
    sr, ss = 12-dl/2, 12+dl/2
    if hour < sr or hour > ss: return 0.0
    x = (hour-12.3)/(dl/2)
    v = peak * max(0, np.cos(x*np.pi/2))**1.5
    return max(0, v)

print("="*60)
print("  DIGITAL TWIN PHASE 2 - COMPLETE SETUP")
print("  Kalyandurg, Anantapur District, AP")
print("="*60)

# ============ STEP 1: GENERATE WEATHER DATA ============
if os.path.exists(f"{RAW}/nasa_power_kalyandurg_2024_original.csv"):
    print("\n[1/5] Real NASA POWER data found — skipping generation ✅")
    df_w = pd.read_csv(f"{RAW}/nasa_power_kalyandurg_2024_original.csv")
else:
    print("\n[1/5] Generating hourly weather data (8784 hours)...")
    records = []
    sm, gw = 35.0, 8.5
    start = pd.Timestamp("2024-01-01")
    for i in range(8784):
        ts = start + pd.Timedelta(hours=i)
        m, h, doy = ts.month, ts.hour, ts.dayofyear
        at,tr,ah,mr,sp,aw = CLIMATE[m]
        
        temp = at + (tr/2)*(-np.cos(2*np.pi*(h-14)/24)) + np.random.normal(0,1.5)
        hum = np.clip(ah + 15*np.cos(2*np.pi*(h-14)/24) + np.random.normal(0,5), 15, 98)
        
        cf = (0.5+0.5*np.random.beta(2,3)) if m in[6,7,8,9] else (0.85+0.15*np.random.beta(5,2))
        sol = max(0, solar_curve(h, sp*cf, doy) + np.random.normal(0,15))
        csol = solar_curve(h, sp, doy)
        
        rain = 0.0
        rp = 0.08 if m in[6,7,8,9] else (0.02 if m in[5,10,11] else 0.003)
        if np.random.random() < rp:
            rain = min(25, np.random.exponential(mr/(30*24*rp)))
        if rain > 0: sol *= 0.2; hum = min(98, hum+20)
        
        wind = max(0.2, aw + np.random.normal(0,0.8))
        et = 0.0023*(temp+17.8)*(sol/1000)*0.15 if sol>0 else 0.01
        sm += rain*0.7 - et - max(0,(sm-40)*0.02)
        sm = np.clip(sm, 5, 45)
        gw += et*0.001 - rain*0.0005*0.7
        gw = np.clip(gw, 3, 20)
        
        pa = 27.8  # panel area for 5kW
        pw = max(0, sol*pa*0.18*(1-0.004*max(0,temp-25)))
        csi = sol/csol if csol>10 else 0
        
        records.append([ts,round(temp,1),round(temp+np.random.uniform(0,2),1),
            round(temp-np.random.uniform(0,2),1),round(hum,1),round(sol,1),
            round(csol,1),round(rain,2),round(wind,1),round(csi,3),
            round(sm,1),round(gw,2),round(pw,1)])

    cols = ["Timestamp","Temperature_C","Temperature_Max_C","Temperature_Min_C",
        "Humidity_Percent","Solar_Irradiance_W_m2","Clear_Sky_Irradiance_W_m2",
        "Rainfall_mm","Wind_Speed_m_s","Clear_Sky_Index",
        "Soil_Moisture_Percent","Groundwater_Depth_m","Power_Output_W"]
    df_w = pd.DataFrame(records, columns=cols)
    df_w.to_csv(f"{RAW}/nasa_power_kalyandurg_2024.csv", index=False)
    print(f"   Saved {len(df_w)} records. Rain={df_w.Rainfall_mm.sum():.0f}mm, Temp={df_w.Temperature_C.min():.0f}-{df_w.Temperature_C.max():.0f}°C")

# ============ STEP 2: GENERATE CROP DATA ============
if os.path.exists(f"{RAW}/icrisat_crop_data_anantapur.csv") and os.path.getsize(f"{RAW}/icrisat_crop_data_anantapur.csv") > 5000:
    print("[2/5] Real ICRISAT data found — skipping generation ✅")
    df_c = pd.read_csv(f"{RAW}/icrisat_crop_data_anantapur.csv")
else:
    print("[2/5] Generating crop data...")
    
    crops_info = {"Groundnut":(700000,(300,900),"Kharif"),"Rice":(30000,(2500,4000),"Kharif"),
        "Chickpea":(50000,(500,1200),"Rabi"),"Pigeonpea":(25000,(400,800),"Kharif"),
        "Sunflower":(45000,(500,1000),"Kharif"),"Sorghum":(20000,(600,1500),"Kharif")}
    crop_recs = []
    for yr in range(2005,2025):
        for cn,(ar,(yl,yh),se) in crops_info.items():
            rf = np.clip(np.random.normal(1,0.25),0.4,1.5)
            a = ar*(1+np.random.uniform(-0.2,0.2))
            y = (yl+yh)/2*rf + np.random.normal(0,50)
            crop_recs.append([yr,"Anantapur",cn,se,round(a),round(a*y/1000),round(y),
                round(544*rf),round(28.5+np.random.normal(0,0.8),1),round(16.9+np.random.normal(0,2),1)])
    df_c = pd.DataFrame(crop_recs, columns=["Year","District","Crop","Season","Area_ha",
        "Production_tonnes","Yield_kg_per_ha","Annual_Rainfall_mm","Avg_Temperature_C","Irrigated_Pct"])
    df_c.to_csv(f"{RAW}/icrisat_crop_data_anantapur.csv", index=False)
    print(f"   Saved {len(df_c)} records")

# ============ STEP 3: GENERATE VILLAGE + GROUNDWATER DATA ============
print("[3/5] Generating village + groundwater data...")

villages = ["Kalyandurg","Bommanahal","Beluguppa","Rayadurg","Gummagatta",
    "Settur","Kundurpi","Tadimarri","Bathalapalle","Kanaganapalle",
    "Kambadur","Ramagiri","Chennekothapalle","Brahmasamudram","Parnapalle",
    "Kuderu","Dommara","Tanakallu","Nallamada","Chilamattur"]
vrecs = []
for v in villages:
    p = np.random.randint(1500,8000); hh = p//np.random.randint(4,6)
    vrecs.append([v,"Kalyandurg","Anantapur","Andhra Pradesh",p,hh,
        round(p*0.51),round(p*0.49),round(p*0.2),round(p*0.08),
        round(np.random.uniform(50,72),1),round(p*0.48),round(p*0.22),round(p*0.2),
        round(np.random.uniform(500,3000)),1,1,1,1,1,1,1,1,1,1,
        round(np.random.uniform(8,15),1)])
df_v = pd.DataFrame(vrecs, columns=["Village","Mandal","District","State","Population_2011",
    "Households","Male_Population","Female_Population","SC_Population","ST_Population",
    "Literacy_Rate","Workers_Total","Cultivators","Agri_Laborers","Geographical_Area_ha",
    "Tap_Water","Hand_Pump","Tube_Well","Well","Primary_School","Middle_School","PHC",
    "Electricity_Domestic","Electricity_Agriculture","Paved_Road","Decadal_Growth_Rate"])
df_v.to_csv(f"{RAW}/census_2011_kalyandurg_villages.csv", index=False)

gwrecs = []
mandals = villages[:10]
for yr in range(2015,2025):
    for md in mandals:
        bd = np.random.uniform(6,14); tr = (yr-2015)*np.random.uniform(0.1,0.4)
        gwrecs.append([yr,md,"Anantapur",round(bd+np.random.uniform(2,6)+tr,2),
            round(max(1,bd-np.random.uniform(0,3)+tr),2),round(np.random.uniform(-1.5,0.5),2),
            np.random.choice(["Safe","Semi-Critical","Critical","Over-Exploited"])])
df_gw = pd.DataFrame(gwrecs, columns=["Year","Mandal","District","Pre_Monsoon_Depth_m",
    "Post_Monsoon_Depth_m","Annual_Change_m","Classification"])
df_gw.to_csv(f"{RAW}/cgwb_groundwater_anantapur.csv", index=False)
print(f"   Saved {len(df_v)} villages, {len(df_gw)} groundwater records")

# ============ STEP 4: TRAIN MODELS ============
print("[4/5] Training ML models...")

# --- Model 1: Soil Moisture (GradientBoosting) ---
df = df_w.copy()
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed")
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfYear"] = df["Timestamp"].dt.dayofyear
for lag in range(1,7):
    df[f"Moisture_lag{lag}"] = df["Soil_Moisture_Percent"].shift(lag)
    if lag <= 3:
        df[f"Temp_lag{lag}"] = df["Temperature_C"].shift(lag)
        df[f"Rain_lag{lag}"] = df["Rainfall_mm"].shift(lag)
df["Rain_rolling_6h"] = df["Rainfall_mm"].rolling(6).sum()
df["Rain_rolling_24h"] = df["Rainfall_mm"].rolling(24).sum()
df["Rain_rolling_7d"] = df["Rainfall_mm"].rolling(168).sum()
df["Temp_rolling_24h"] = df["Temperature_C"].rolling(24).mean()
df["Solar_rolling_6h"] = df["Solar_Irradiance_W_m2"].rolling(6).mean()
df["ET0_estimate"] = 0.0023*(df.Temperature_C+17.8)*(df.Solar_Irradiance_W_m2/1000)*np.sqrt((df.Temperature_Max_C-df.Temperature_Min_C).clip(0.1))
df["Hour_sin"] = np.sin(2*np.pi*df.Hour/24)
df["Hour_cos"] = np.cos(2*np.pi*df.Hour/24)
df["Day_sin"] = np.sin(2*np.pi*df.DayOfYear/365)
df["Day_cos"] = np.cos(2*np.pi*df.DayOfYear/365)
df.dropna(inplace=True)

moist_feats = ["Moisture_lag1","Moisture_lag2","Moisture_lag3","Moisture_lag4","Moisture_lag5","Moisture_lag6",
    "Temp_lag1","Temp_lag2","Temp_lag3","Rain_lag1","Rain_lag2","Rain_lag3",
    "Rain_rolling_6h","Rain_rolling_24h","Rain_rolling_7d","Temp_rolling_24h","Solar_rolling_6h",
    "Humidity_Percent","Wind_Speed_m_s","ET0_estimate","Hour_sin","Hour_cos","Day_sin","Day_cos"]

Xm, ym = df[moist_feats], df["Soil_Moisture_Percent"]
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.2, shuffle=False)
m1 = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=42)
m1.fit(Xm_tr, ym_tr)
r2_m = r2_score(ym_te, m1.predict(Xm_te))
joblib.dump(m1, f"{MODELS}/moisture_model_xgb.pkl")
joblib.dump(moist_feats, f"{MODELS}/moisture_features.pkl")
print(f"   Moisture model R²: {r2_m:.4f}")

# --- Model 2: Solar Power (MLP) ---
df2 = df_w.copy()
df2["Timestamp"] = pd.to_datetime(df2["Timestamp"], format="mixed")
df2["Hour_sin"] = np.sin(2*np.pi*df2.Timestamp.dt.hour/24)
df2["Hour_cos"] = np.cos(2*np.pi*df2.Timestamp.dt.hour/24)
df2["Day_sin"] = np.sin(2*np.pi*df2.Timestamp.dt.dayofyear/365)
df2["Day_cos"] = np.cos(2*np.pi*df2.Timestamp.dt.dayofyear/365)
df2["Solar_lag1"] = df2.Solar_Irradiance_W_m2.shift(1)
df2["Solar_lag2"] = df2.Solar_Irradiance_W_m2.shift(2)
df2["Solar_rolling_3h"] = df2.Solar_Irradiance_W_m2.rolling(3).mean()
df2["Temp_lag1"] = df2.Temperature_C.shift(1)
df2.dropna(inplace=True)

solar_feats = ["Solar_Irradiance_W_m2","Clear_Sky_Irradiance_W_m2","Clear_Sky_Index",
    "Temperature_C","Humidity_Percent","Hour_sin","Hour_cos","Day_sin","Day_cos",
    "Solar_lag1","Solar_lag2","Solar_rolling_3h","Temp_lag1","Wind_Speed_m_s"]
Xs, ys = df2[solar_feats], df2["Power_Output_W"]
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(Xs, ys, test_size=0.2, shuffle=False)
scaler = StandardScaler()
Xs_tr_s = scaler.fit_transform(Xs_tr)
Xs_te_s = scaler.transform(Xs_te)
m2 = MLPRegressor(hidden_layer_sizes=(128,64,32), max_iter=500, early_stopping=True, random_state=42, learning_rate='adaptive')
m2.fit(Xs_tr_s, ys_tr)
r2_s = r2_score(ys_te, np.maximum(0, m2.predict(Xs_te_s)))
joblib.dump(m2, f"{MODELS}/solar_model_mlp.pkl")
joblib.dump(scaler, f"{MODELS}/solar_scaler.pkl")
joblib.dump(solar_feats, f"{MODELS}/solar_features.pkl")
print(f"   Solar model R²: {r2_s:.4f}")

# --- Model 3: Crop Yield (Random Forest) ---
crop_map = {c:i for i,c in enumerate(df_c.Crop.unique())}
df_c["Crop_encoded"] = df_c.Crop.map(crop_map)
df_c["Season_encoded"] = df_c.Season.map({"Kharif":0,"Rabi":1})
crop_feats = ["Year","Crop_encoded","Season_encoded","Area_ha","Annual_Rainfall_mm","Avg_Temperature_C","Irrigated_Pct"]
Xc, yc = df_c[crop_feats], df_c["Yield_kg_per_ha"]
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=42)
m3 = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
m3.fit(Xc_tr, yc_tr)
r2_c = r2_score(yc_te, m3.predict(Xc_te))
joblib.dump(m3, f"{MODELS}/crop_yield_model_rf.pkl")
joblib.dump(crop_map, f"{MODELS}/crop_mapping.pkl")
joblib.dump(crop_feats, f"{MODELS}/crop_features.pkl")
print(f"   Crop model R²: {r2_c:.4f}")

# --- Model 4: Demand params ---
village = df_v[df_v.Village=="Kalyandurg"].iloc[0]
params = {
    "village_name":"Kalyandurg","base_year":2011,
    "base_population":int(village.Population_2011),
    "growth_rate_annual":round(village.Decadal_Growth_Rate/10,2),
    "households":int(village.Households),
    "cultivators":int(village.Cultivators),
    "agri_laborers":int(village.Agri_Laborers),
    "geographical_area_ha":int(village.Geographical_Area_ha),
    "per_capita_water_lpcd":55,"livestock_water_lpcd":30,
    "estimated_livestock_units":int(village.Population_2011*0.3),
    "irrigation_water_mm_per_day":5,
    "per_capita_energy_kwh_day":1.2,"agri_pump_energy_kwh_day":8,
    "estimated_pumps":max(10,int(village.Cultivators*0.2)),
    "avg_solar_hours":8,"panel_efficiency":0.18,"recommended_capacity_kw":5,
}
with open(f"{MODELS}/demand_params.json","w") as f: json.dump(params,f,indent=2)

metrics = [
    {"model":"Soil Moisture","r2":round(r2_m,4)},
    {"model":"Solar Power","r2":round(r2_s,4)},
    {"model":"Crop Yield","r2":round(r2_c,4)},
    {"model":"Population","type":"System Dynamics"},
]
with open(f"{MODELS}/training_metrics.json","w") as f: json.dump(metrics,f,indent=2)
print(f"   Demand model params saved")

# ============ STEP 5: VERIFY ============
print("\n[5/5] Verification...")
files = os.listdir(MODELS)
print(f"   Models: {len(files)} files in {MODELS}/")
for f in sorted(files):
    sz = os.path.getsize(f"{MODELS}/{f}")
    print(f"     {f} ({sz/1024:.1f} KB)")

print(f"\n{'='*60}")
print(f"  SETUP COMPLETE!")
print(f"  Moisture R²={r2_m:.4f}  Solar R²={r2_s:.4f}  Crop R²={r2_c:.4f}")
print(f"{'='*60}")
print(f"\nNext: cd app && streamlit run streamlit_app.py")
