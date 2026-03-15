"""
Convert REAL ICRISAT District-Level crop data to project format.

USAGE:
    python convert_icrisat_data.py icrisat_crop_data_anantapur_orig.csv

The ICRISAT data has WIDE format (one row per year, crops as columns):
    RICE AREA | RICE PROD | RICE YIELD | SORGHUM AREA | SORGHUM PROD | ...

We convert to LONG format (one row per crop per year):
    Year | Crop | Area_ha | Production_tonnes | Yield_kg_per_ha | ...
"""

import pandas as pd
import numpy as np
import sys
import os


def convert_icrisat(input_file, output_file=None):
    print(f"Reading: {input_file}")
    df = pd.read_csv(input_file)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows: {len(df)}")
    
    # Identify crops from column names
    # Pattern: CROP_AREA, CROP_PROD*, CROP_YIELD*
    # The columns look like: RICE AREA, RICE PROD, RICE YIELD, SORGHUM AREA, etc.
    
    # Define the crops we expect and their column name patterns
    crop_configs = {
        "Rice": "RICE",
        "Sorghum": "SORGHUM",
        "Chickpea": "CHICKPEA",
        "Pigeonpea": "PIGEONPE",  # truncated in ICRISAT
        "Groundnut": "GROUNDN",   # truncated in ICRISAT  
        "Sunflower": "SUNFLOW",   # truncated in ICRISAT
    }
    
    records = []
    
    for _, row in df.iterrows():
        year = row.get("Year", None)
        if pd.isna(year):
            continue
        year = int(year)
        
        for crop_name, col_prefix in crop_configs.items():
            # Find matching columns (area, production, yield)
            area_col = None
            prod_col = None
            yield_col = None
            
            for col in df.columns:
                col_upper = col.upper()
                prefix_upper = col_prefix.upper()
                
                if prefix_upper in col_upper:
                    if "AREA" in col_upper:
                        area_col = col
                    elif "PROD" in col_upper:
                        prod_col = col
                    elif "YIELD" in col_upper:
                        yield_col = col
            
            if area_col is None and prod_col is None:
                continue
            
            area = row.get(area_col, 0) if area_col else 0
            prod = row.get(prod_col, 0) if prod_col else 0
            yld = row.get(yield_col, 0) if yield_col else 0
            
            # Convert to numeric, handle NaN/0
            area = pd.to_numeric(area, errors='coerce')
            prod = pd.to_numeric(prod, errors='coerce')
            yld = pd.to_numeric(yld, errors='coerce')
            
            if pd.isna(area) or area <= 0:
                continue
            
            # ICRISAT units: Area in '000 hectares, Production in '000 tonnes
            # Yield in kg/ha
            area_ha = area * 1000
            prod_tonnes = prod * 1000 if not pd.isna(prod) else 0
            yield_kg = yld if not pd.isna(yld) else (prod_tonnes * 1000 / area_ha if area_ha > 0 else 0)
            
            # Assign season
            season = "Rabi" if crop_name == "Chickpea" else "Kharif"
            
            # Estimate weather from year (will be overridden by NASA data later)
            # Anantapur avg rainfall ~544mm, temp ~28.5°C
            avg_rain = 544 + np.random.normal(0, 80)
            avg_temp = 28.5 + np.random.normal(0, 0.8)
            irr_pct = 16.9 + np.random.normal(0, 2)
            
            records.append({
                "Year": year,
                "District": "Anantapur",
                "Crop": crop_name,
                "Season": season,
                "Area_ha": round(area_ha),
                "Production_tonnes": round(prod_tonnes),
                "Yield_kg_per_ha": round(yield_kg, 1),
                "Annual_Rainfall_mm": round(avg_rain),
                "Avg_Temperature_C": round(avg_temp, 1),
                "Irrigated_Pct": round(max(5, irr_pct), 1),
            })
    
    df_out = pd.DataFrame(records)
    
    # Remove rows with 0 yield
    df_out = df_out[df_out["Yield_kg_per_ha"] > 0].reset_index(drop=True)
    
    # Save
    if output_file is None:
        output_file = os.path.join(
            os.path.dirname(input_file), "icrisat_crop_data_anantapur.csv"
        )
    
    df_out.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"  CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"  Output: {output_file}")
    print(f"  Records: {len(df_out)}")
    print(f"  Year range: {df_out['Year'].min()} - {df_out['Year'].max()}")
    print(f"\n  Crops found:")
    for crop, group in df_out.groupby("Crop"):
        print(f"    {crop:<15} {len(group)} years | "
              f"Avg yield: {group['Yield_kg_per_ha'].mean():.0f} kg/ha | "
              f"Avg area: {group['Area_ha'].mean()/1000:.0f}K ha")
    
    print(f"\n✅ Ready! Now run: python setup_all.py")
    return df_out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_icrisat_data.py <input_csv>")
        sys.exit(1)
    
    if not os.path.exists(sys.argv[1]):
        print(f"ERROR: File not found: {sys.argv[1]}")
        sys.exit(1)
    
    convert_icrisat(sys.argv[1])
