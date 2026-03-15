"""
Convert REAL NASA POWER CSV to the format our Digital Twin models expect.

USAGE:
    python convert_nasa_data.py nasa_power_kalyandurg_2024_orig.csv

This will create: nasa_power_kalyandurg_2024.csv (ready for the project)

NASA POWER columns → Our columns:
    YEAR, MO, DY, HR          → Timestamp
    T2M (°C)                  → Temperature_C
    T2M (derived)             → Temperature_Max_C, Temperature_Min_C
    RH2M (%)                  → Humidity_Percent
    ALLSKY_SFC_SW_DWN (MJ/hr) → Solar_Irradiance_W_m2  (converted: MJ/hr × 277.78 = W/m²)
    CLRSKY_SFC_SW_DWN (MJ/hr) → Clear_Sky_Irradiance_W_m2
    PRECTOTCORR (mm/hour)     → Rainfall_mm
    WS2M (m/s)                → Wind_Speed_m_s
    Derived                   → Clear_Sky_Index, Soil_Moisture_Percent, 
                                 Groundwater_Depth_m, Power_Output_W
"""

import pandas as pd
import numpy as np
import sys
import os


def convert_nasa_power(input_file, output_file=None):
    """Convert NASA POWER CSV to project format"""
    
    print(f"Reading: {input_file}")
    
    # NASA POWER CSVs have header rows before the actual data
    # Find the row where actual data starts (look for "YEAR" header)
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    header_row = None
    for i, line in enumerate(lines):
        if line.strip().startswith("YEAR"):
            header_row = i
            break
    
    if header_row is None:
        print("ERROR: Could not find 'YEAR' header row. Is this a NASA POWER CSV?")
        return
    
    print(f"  Found data header at line {header_row + 1}")
    
    # Read the CSV, skipping the metadata header
    df = pd.read_csv(input_file, skiprows=header_row)
    
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    print(f"  Columns found: {list(df.columns)}")
    print(f"  Raw rows: {len(df)}")
    
    # --- Build Timestamp ---
    df["Timestamp"] = pd.to_datetime(
        df["YEAR"].astype(str) + "-" + 
        df["MO"].astype(str).str.zfill(2) + "-" + 
        df["DY"].astype(str).str.zfill(2) + " " + 
        df["HR"].astype(str).str.zfill(2) + ":00:00"
    )
    
    # --- Replace -999 (NASA missing value) with NaN ---
    df = df.replace(-999, np.nan)
    df = df.replace(-999.0, np.nan)
    
    # --- Convert Solar Irradiance: MJ/hr → W/m² ---
    # 1 MJ/hr = 1,000,000 J / 3600 s = 277.78 W/m²
    MJ_TO_W = 277.78
    
    df["Solar_Irradiance_W_m2"] = (df["ALLSKY_SFC_SW_DWN"] * MJ_TO_W).round(1).clip(lower=0)
    df["Clear_Sky_Irradiance_W_m2"] = (df["CLRSKY_SFC_SW_DWN"] * MJ_TO_W).round(1).clip(lower=0)
    
    # --- Direct mappings ---
    df["Temperature_C"] = df["T2M"].round(1)
    df["Humidity_Percent"] = df["RH2M"].round(1)
    df["Rainfall_mm"] = df["PRECTOTCORR"].round(2).clip(lower=0)
    df["Wind_Speed_m_s"] = df["WS2M"].round(1)
    
    # --- Derived: Daily max/min temperature (rolling 24h window) ---
    df["Temperature_Max_C"] = df["Temperature_C"].rolling(24, min_periods=1).max().round(1)
    df["Temperature_Min_C"] = df["Temperature_C"].rolling(24, min_periods=1).min().round(1)
    
    # --- Clear Sky Index ---
    df["Clear_Sky_Index"] = np.where(
        df["Clear_Sky_Irradiance_W_m2"] > 10,
        (df["Solar_Irradiance_W_m2"] / df["Clear_Sky_Irradiance_W_m2"]).clip(0, 1.2),
        0
    ).round(3)
    
    # --- Simulated Soil Moisture (water balance model for red sandy loam) ---
    # This is a physics-based estimate until real soil sensors are available
    soil_moisture = 30.0  # Initial value (% volumetric)
    groundwater = 9.0     # meters below ground
    moisture_vals = []
    gw_vals = []
    
    for _, row in df.iterrows():
        rain = row["Rainfall_mm"] if not np.isnan(row["Rainfall_mm"]) else 0
        solar = row["Solar_Irradiance_W_m2"] if not np.isnan(row["Solar_Irradiance_W_m2"]) else 0
        temp = row["Temperature_C"] if not np.isnan(row["Temperature_C"]) else 28
        
        # Infiltration
        infiltration = rain * 0.7  # 70% infiltrates in sandy loam
        
        # ET (simplified Hargreaves)
        et = 0.0023 * (temp + 17.8) * (solar / 1000) * 0.15 if solar > 0 else 0.01
        
        # Drainage
        drainage = max(0, (soil_moisture - 40) * 0.02)
        
        soil_moisture += infiltration - et - drainage
        soil_moisture = np.clip(soil_moisture, 5, 45)
        
        groundwater += et * 0.001 - infiltration * 0.0005
        groundwater = np.clip(groundwater, 3, 20)
        
        moisture_vals.append(round(soil_moisture, 1))
        gw_vals.append(round(groundwater, 2))
    
    df["Soil_Moisture_Percent"] = moisture_vals
    df["Groundwater_Depth_m"] = gw_vals
    
    # --- Solar Power Output (5kW system) ---
    panel_area = 27.8  # m² for 5kW at 18% efficiency
    efficiency = 0.18
    temp_coeff = -0.004  # -0.4%/°C above 25°C
    
    df["Power_Output_W"] = (
        df["Solar_Irradiance_W_m2"] * panel_area * efficiency * 
        (1 + temp_coeff * (df["Temperature_C"] - 25).clip(lower=0))
    ).clip(lower=0).round(1)
    
    # --- Select final columns ---
    output_cols = [
        "Timestamp", "Temperature_C", "Temperature_Max_C", "Temperature_Min_C",
        "Humidity_Percent", "Solar_Irradiance_W_m2", "Clear_Sky_Irradiance_W_m2",
        "Rainfall_mm", "Wind_Speed_m_s", "Clear_Sky_Index",
        "Soil_Moisture_Percent", "Groundwater_Depth_m", "Power_Output_W"
    ]
    
    df_out = df[output_cols].copy()
    
    # --- Handle remaining NaN ---
    nan_count = df_out.isnull().sum().sum()
    if nan_count > 0:
        print(f"  Filling {nan_count} NaN values with forward-fill + backfill...")
        df_out = df_out.ffill().bfill()
    
    # --- Save ---
    if output_file is None:
        output_file = os.path.join(os.path.dirname(input_file), "nasa_power_kalyandurg_2024_original.csv")
    
    df_out.to_csv(output_file, index=False)
    
    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"  CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"  Output: {output_file}")
    print(f"  Records: {len(df_out)}")
    print(f"  Date range: {df_out['Timestamp'].iloc[0]} to {df_out['Timestamp'].iloc[-1]}")
    print(f"\n  Column Summary:")
    print(f"  {'Column':<30} {'Min':>8} {'Max':>8} {'Mean':>8}")
    print(f"  {'-'*56}")
    for col in output_cols[1:]:  # skip Timestamp
        print(f"  {col:<30} {df_out[col].min():>8.1f} {df_out[col].max():>8.1f} {df_out[col].mean():>8.1f}")
    
    print(f"\n  Total rainfall: {df_out['Rainfall_mm'].sum():.1f} mm")
    print(f"  Avg solar peak: {df_out[df_out['Solar_Irradiance_W_m2']>0]['Solar_Irradiance_W_m2'].mean():.1f} W/m²")
    print(f"\n✅ Now re-run: python setup_all.py  (to retrain models with real data)")
    
    return df_out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_nasa_data.py <input_csv_path>")
        print("Example: python convert_nasa_data.py nasa_power_kalyandurg_2024_original.csv")
        sys.exit(1)
    
    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)
    
    convert_nasa_power(input_path)
