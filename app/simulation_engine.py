"""Digital Twin Simulation Engine - Core 30-day forward simulation"""
import numpy as np, pandas as pd, joblib, json, os

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")

class DigitalTwinSimulator:
    def __init__(self):
        self.moisture_model = joblib.load(f"{MODEL_DIR}/moisture_model_xgb.pkl")
        self.moisture_features = joblib.load(f"{MODEL_DIR}/moisture_features.pkl")
        self.solar_model = joblib.load(f"{MODEL_DIR}/solar_model_mlp.pkl")
        self.solar_scaler = joblib.load(f"{MODEL_DIR}/solar_scaler.pkl")
        self.solar_features = joblib.load(f"{MODEL_DIR}/solar_features.pkl")
        self.crop_model = joblib.load(f"{MODEL_DIR}/crop_yield_model_rf.pkl")
        self.crop_mapping = joblib.load(f"{MODEL_DIR}/crop_mapping.pkl")
        self.weather_df = pd.read_csv(f"{DATA_DIR}/nasa_power_kalyandurg_2024.csv")
        self.weather_df["Timestamp"] = pd.to_datetime(self.weather_df["Timestamp"])
        with open(f"{MODEL_DIR}/demand_params.json") as f: self.demand_params = json.load(f)

    def predict_crop_yield(self, crop_name, rainfall_mm, temperature_c, irrigated_pct=16.9):
        if crop_name not in self.crop_mapping: return None
        X = pd.DataFrame([{"Year":2025,"Crop_encoded":self.crop_mapping[crop_name],
            "Season_encoded":0,"Area_ha":50000,"Annual_Rainfall_mm":rainfall_mm,
            "Avg_Temperature_C":temperature_c,"Irrigated_Pct":irrigated_pct}])
        return float(self.crop_model.predict(X)[0])

    def calculate_demand(self, year, month, growth_rate_override=None):
        p = self.demand_params; gr = growth_rate_override or p["growth_rate_annual"]
        pop = p["base_population"]*(1+gr/100)**(year-p["base_year"])
        dw = pop*p["per_capita_water_lpcd"]
        f = 0.6 if month in[6,7,8,9] else (0.2 if month in[10,11,12,1] else 0.1)
        aw = p["geographical_area_ha"]*f*p["irrigation_water_mm_per_day"]*10
        de = pop*p["per_capita_energy_kwh_day"]
        ae = p["estimated_pumps"]*p["agri_pump_energy_kwh_day"]
        return {"population":round(pop),"domestic_water_lpd":round(dw),"agri_water_lpd":round(aw),
            "total_water_lpd":round(dw+aw),"total_energy_kwh":round(de+ae,1)}

    def run_simulation(self, days=30, start_month=6, start_year=2025,
                       solar_capacity_kw=5, rainfall_deviation=0,
                       population_growth_override=None, crop="Groundnut"):
        start_date = pd.Timestamp(f"{start_year}-{start_month:02d}-01")
        sm, gw, cum_rain, cum_irr = 25.0, 9.0, 0, 0
        ps = solar_capacity_kw/5.0
        PUMP_W, CRIT, WARN = 1500, 20, 28
        results = []

        for day in range(days):
            cd = start_date + pd.Timedelta(days=day)
            month = cd.month
            mask = self.weather_df["Timestamp"].dt.month == month
            md = self.weather_df[mask]
            avail_days = md["Timestamp"].dt.date.unique()
            sd = np.random.choice(avail_days)
            dw = md[md["Timestamp"].dt.date == sd].head(24).reset_index(drop=True)
            if rainfall_deviation:
                dw["Rainfall_mm"] = (dw["Rainfall_mm"]*(1+rainfall_deviation/100)).clip(0)
            
            dp, dr, di = 0, dw["Rainfall_mm"].sum(), 0
            max_pw = 0
            for _, r in dw.iterrows():
                sol = r.get("Solar_Irradiance_W_m2",0)
                pw = max(0, sol*27.8*0.18*(1-0.004*max(0,r.get("Temperature_C",28)-25)))*ps
                dp += pw/1000; max_pw = max(max_pw, pw)
                et = 0.0023*(r.Temperature_C+17.8)*(sol/1000)*0.15 if sol>0 else 0.01
                sm += r.Rainfall_mm*0.7 - et - max(0,(sm-40)*0.02)
                sm = np.clip(sm, 5, 45)
                if sm < CRIT and pw > PUMP_W: sm += 2; di += 1; gw += 0.01
                gw -= r.Rainfall_mm*0.0005; gw = np.clip(gw, 3, 20)
            
            cum_rain += dr; cum_irr += di
            at = dw.Temperature_C.mean(); ah = dw.Humidity_Percent.mean()
            demand = self.calculate_demand(start_year, month, population_growth_override)
            pr = cum_rain*(365/max(1,day+1))
            cy = self.predict_crop_yield(crop, pr, at)
            
            if sm < CRIT:
                st = "IRRIGATE_SOLAR" if max_pw > PUMP_W else "IRRIGATE_WARNING"
                sc = "orange" if max_pw > PUMP_W else "red"
            elif sm < WARN: st, sc = "MONITOR", "yellow"
            else: st, sc = "OPTIMAL", "green"
            
            results.append({"day":day+1,"date":cd.strftime("%Y-%m-%d"),"month":month,
                "avg_temperature":round(at,1),"avg_humidity":round(ah,1),
                "rainfall_mm":round(dr,1),"cumulative_rain_mm":round(cum_rain,1),
                "soil_moisture":round(sm,1),"groundwater_depth_m":round(gw,2),
                "power_generated_kwh":round(dp,2),"peak_power_w":round(max_pw,1),
                "energy_demand_kwh":demand["total_energy_kwh"],
                "energy_surplus_kwh":round(dp-demand["total_energy_kwh"],2),
                "projected_yield_kg_ha":round(cy,1) if cy else None,
                "population":demand["population"],"water_demand_lpd":demand["total_water_lpd"],
                "water_from_rain_lpd":round(dr*10*self.demand_params.get("geographical_area_ha",1000)),
                "irrigation_hours":di,"cumulative_irrigation_hours":cum_irr,
                "status":st,"status_text":st,"status_color":sc})
        return results

    def get_summary(self, results):
        df = pd.DataFrame(results)
        return {"total_days":len(results),"avg_soil_moisture":round(df.soil_moisture.mean(),1),
            "min_soil_moisture":round(df.soil_moisture.min(),1),
            "total_rainfall_mm":round(df.rainfall_mm.sum(),1),
            "total_power_kwh":round(df.power_generated_kwh.sum(),1),
            "total_energy_demand_kwh":round(df.energy_demand_kwh.sum(),1),
            "energy_self_sufficiency_pct":round(df.power_generated_kwh.sum()/max(1,df.energy_demand_kwh.sum())*100,1),
            "days_irrigation_needed":int((df.irrigation_hours>0).sum()),
            "total_irrigation_hours":int(df.irrigation_hours.sum()),
            "projected_crop_yield":round(df.projected_yield_kg_ha.iloc[-1],1) if df.projected_yield_kg_ha.iloc[-1] else None,
            "status_breakdown":df.status.value_counts().to_dict(),
            "avg_groundwater_depth":round(df.groundwater_depth_m.mean(),2)}

if __name__ == "__main__":
    sim = DigitalTwinSimulator()
    results = sim.run_simulation(days=30, start_month=6)
    s = sim.get_summary(results)
    print("Simulation Summary:")
    for k,v in s.items(): print(f"  {k}: {v}")
    print("\nTimeline:")
    for r in results:
        b = {"OPTIMAL":"G","MONITOR":"Y","IRRIGATE_SOLAR":"O","IRRIGATE_WARNING":"R"}.get(r["status"],"?")
        print(f"  Day {r['day']:2d} [{b}] SM:{r['soil_moisture']:5.1f}% Rain:{r['rainfall_mm']:5.1f}mm Pow:{r['power_generated_kwh']:6.1f}kWh GW:{r['groundwater_depth_m']:5.2f}m")
