"""
Digital Twin Dashboard - Kalyandurg Village
WITH RAG Chatbot Integration
Run: cd app && python -m streamlit run streamlit_app.py
"""
import streamlit as st
import pandas as pd, numpy as np, json, sys, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation_engine import DigitalTwinSimulator

st.set_page_config(page_title="Digital Twin - Kalyandurg", page_icon="🌾", layout="wide")

@st.cache_resource
def load_sim(): return DigitalTwinSimulator()
sim = load_sim()

# --- RAG Setup ---
@st.cache_resource
def load_rag(api_key):
    """Initialize RAG system (cached so it only runs once)"""
    try:
        from rag_system import AgriRAG
        rag = AgriRAG(api_key=api_key)
        success = rag.initialize()
        if success:
            return rag
        return None
    except Exception as e:
        st.error(f"RAG init error: {e}")
        return None

# Sidebar
st.sidebar.markdown("## 🎛️ Scenario Controls")
solar_kw = st.sidebar.slider("☀️ Solar Capacity (kW)", 1, 20, 5)
rain_dev = st.sidebar.slider("🌧️ Rainfall Deviation (%)", -50, 50, 0, 5)
pop_gr = st.sidebar.slider("👥 Pop Growth (%/yr)", 0.0, 3.0, 1.2, 0.1)
start_mo = st.sidebar.selectbox("📅 Start Month", range(1,13),
    format_func=lambda x: pd.Timestamp(2025,x,1).strftime("%B"), index=5)
sim_days = st.sidebar.slider("📆 Days", 7, 90, 30, 7)
crop = st.sidebar.selectbox("🌱 Crop", ["Groundnut","Rice","Chickpea","Sunflower","Sorghum","Pigeonpea"])
run = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

# Gemini API key in sidebar
st.sidebar.divider()
st.sidebar.markdown("### 🤖 AI Advisor Setup")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password", 
    help="Get from https://aistudio.google.com/apikey")

st.markdown("<h1 style='text-align:center;color:#1B5E20'>🌾 Digital Twin — Kalyandurg Village</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#666'>Anantapur District, AP | Multi-Domain Simulator</p>", unsafe_allow_html=True)

if "results" not in st.session_state or run:
    with st.spinner("Running simulation..."):
        st.session_state.results = sim.run_simulation(sim_days, start_mo, 2025, solar_kw, rain_dev, pop_gr, crop)
        st.session_state.summary = sim.get_summary(st.session_state.results)

results = st.session_state.results
summary = st.session_state.summary
df = pd.DataFrame(results)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Dashboard","💧 Water","⚡ Energy","🌱 Agriculture","💬 AI Advisor","🌐 3D Twin"])

# =================== TAB 1: DASHBOARD ===================
with tab1:
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Avg Moisture", f"{summary['avg_soil_moisture']}%")
    c2.metric("Total Rain", f"{summary['total_rainfall_mm']}mm")
    c3.metric("Solar Gen", f"{summary['total_power_kwh']}kWh")
    c4.metric("Irrigation Days", f"{summary['days_irrigation_needed']}/{sim_days}")
    c5.metric("Yield", f"{summary.get('projected_crop_yield','N/A')} kg/ha")
    
    st.divider()
    colors = {"OPTIMAL":"#4CAF50","MONITOR":"#FFC107","IRRIGATE_SOLAR":"#FF9800","IRRIGATE_WARNING":"#F44336"}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.date, y=df.soil_moisture, mode="lines+markers",
        name="Soil Moisture (%)", line=dict(color="#2196F3", width=2),
        marker=dict(color=[colors.get(s,"#999") for s in df.status], size=8)))
    fig.add_trace(go.Bar(x=df.date, y=df.rainfall_mm, name="Rainfall (mm)",
        yaxis="y2", marker_color="rgba(33,150,243,0.3)"))
    fig.update_layout(yaxis=dict(title="Moisture %",range=[0,50]),
        yaxis2=dict(title="Rain mm",overlaying="y",side="right",range=[0,30]),
        height=400, template="plotly_white", hovermode="x unified")
    fig.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="Critical")
    st.plotly_chart(fig, use_container_width=True)

# =================== TAB 2: WATER ===================
with tab2:
    c1,c2,c3 = st.columns(3)
    c1.metric("Groundwater", f"{summary['avg_groundwater_depth']}m")
    c2.metric("Irrigation Hrs", f"{summary['total_irrigation_hours']}")
    c3.metric("Water Demand", f"{df.water_demand_lpd.mean()/1000:.0f} KL/day")
    fig2 = make_subplots(specs=[[{"secondary_y":True}]])
    fig2.add_trace(go.Scatter(x=df.date,y=df.soil_moisture,name="Moisture %",
        line=dict(color="#795548",width=2),fill="tozeroy",fillcolor="rgba(121,85,72,0.1)"))
    fig2.add_trace(go.Scatter(x=df.date,y=df.groundwater_depth_m,name="GW Depth (m)",
        line=dict(color="#0D47A1",width=2,dash="dot")),secondary_y=True)
    fig2.update_yaxes(title_text="Moisture %",secondary_y=False)
    fig2.update_yaxes(title_text="GW Depth (m)",secondary_y=True,autorange="reversed")
    fig2.update_layout(height=400,template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

# =================== TAB 3: ENERGY ===================
with tab3:
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Gen", f"{summary['total_power_kwh']:.0f} kWh")
    c2.metric("Avg Daily", f"{summary['total_power_kwh']/sim_days:.1f} kWh/day")
    c3.metric("Self-Sufficiency", f"{min(100,summary['energy_self_sufficiency_pct']):.1f}%")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df.date,y=df.power_generated_kwh,name="Generated kWh",
        line=dict(color="#FFC107",width=2),fill="tozeroy",fillcolor="rgba(255,193,7,0.2)"))
    fig3.update_layout(height=350,template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

# =================== TAB 4: AGRICULTURE ===================
with tab4:
    c1,c2 = st.columns(2)
    with c1:
        st.metric(f"{crop} Yield", f"{summary.get('projected_crop_yield','N/A')} kg/ha")
    with c2:
        rains = np.linspace(200,800,20)
        ylds = [sim.predict_crop_yield(crop,r,29) or 0 for r in rains]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=rains,y=ylds,mode="lines+markers",line=dict(color="#4CAF50")))
        fig4.add_vline(x=544,line_dash="dash",annotation_text="Avg Rain")
        fig4.update_layout(xaxis_title="Rainfall (mm)",yaxis_title="Yield (kg/ha)",height=300,template="plotly_white")
        st.plotly_chart(fig4, use_container_width=True)
    
    crops_list = ["Groundnut","Rice","Chickpea","Sunflower","Sorghum","Pigeonpea"]
    pr = df.cumulative_rain_mm.iloc[-1]*365/sim_days
    cy = {c:(sim.predict_crop_yield(c,pr,29) or 0) for c in crops_list}
    fig5 = go.Figure(data=[go.Bar(x=list(cy.keys()),y=list(cy.values()),
        marker_color=["#4CAF50" if c==crop else "#90CAF9" for c in cy])])
    fig5.update_layout(yaxis_title="Yield (kg/ha)",height=350,template="plotly_white")
    st.plotly_chart(fig5, use_container_width=True)

# =================== TAB 5: AI ADVISOR (RAG CHATBOT) ===================
with tab5:
    st.subheader("💬 AI Agricultural Advisor")
    st.markdown("*Powered by Gemini + RAG | Ask about irrigation, solar schemes, crop management, government programs*")
    
    if not gemini_key:
        st.warning("👈 Enter your Gemini API key in the sidebar to activate the AI Advisor")
        st.markdown("""
        **How to get a key:**
        1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
        2. Click "Create API Key"
        3. Paste it in the sidebar
        """)
    else:
        # Initialize RAG
        rag = load_rag(gemini_key)
        
        if rag is None:
            st.error("Failed to initialize RAG system. Check your API key and internet connection.")
        else:
            st.success("✅ AI Advisor ready! Knowledge base loaded with 16 documents.")
            
            # Build simulation context for the chatbot
            sim_context = {
                "soil_moisture": df.soil_moisture.iloc[-1],
                "groundwater_depth": summary["avg_groundwater_depth"],
                "solar_capacity": solar_kw,
                "daily_power": round(summary["total_power_kwh"] / sim_days, 1),
                "total_rain": summary["total_rainfall_mm"],
                "yield": summary.get("projected_crop_yield", "N/A"),
                "crop": crop,
                "population": df.population.iloc[-1],
                "irrigation_status": f"{summary['days_irrigation_needed']} days needed out of {sim_days}",
            }
            
            # Show context in expander
            with st.expander("📋 Simulation context being sent to AI"):
                st.json(sim_context)
            
            # Quick question buttons
            st.markdown("**Quick questions:**")
            qcol1, qcol2, qcol3 = st.columns(3)
            
            quick_q = None
            with qcol1:
                if st.button("🌱 Should I irrigate today?", use_container_width=True):
                    quick_q = "Based on the current soil moisture and weather conditions, should I irrigate my crop today? What are the risks if I don't?"
            with qcol2:
                if st.button("☀️ How to get solar pump subsidy?", use_container_width=True):
                    quick_q = "I want to install a solar pump for irrigation. What government subsidies are available under PM-KUSUM? What is the process and cost?"
            with qcol3:
                if st.button("📊 What crop should I grow?", use_container_width=True):
                    quick_q = f"Given the current rainfall conditions and soil moisture, should I continue growing {crop} or switch to another crop? Compare yield potential of different crops."
            
            st.divider()
            
            # Chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat history
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            # Handle input (either quick question or typed)
            user_input = quick_q or st.chat_input("Ask anything about farming, irrigation, solar, schemes...")
            
            if user_input:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # Generate RAG response
                with st.chat_message("assistant"):
                    with st.spinner("Searching knowledge base & generating answer..."):
                        response = rag.query(user_input, sim_context)
                        st.markdown(response)
                        
                        # Show sources
                        sources = rag.get_sources(user_input)
                        if sources:
                            with st.expander("📚 Sources used"):
                                for title, score in sources:
                                    st.markdown(f"- **{title}** (relevance: {score:.2f})")
                
                st.session_state.messages.append({"role": "assistant", "content": response})

# =================== TAB 6: 3D VISUALIZATION ===================
with tab6:
    st.subheader("🌐 3D Digital Twin - Kalyandurg Village")
    import streamlit.components.v1 as components
    with open("visualizer_3d.html", "r", encoding="utf-8") as f:
        components.html(f.read(), height=700, scrolling=False)


# =================== SIDEBAR FOOTER ===================
st.sidebar.divider()
st.sidebar.markdown("📍 **Kalyandurg** | Anantapur, AP | 14.75°N, 77.11°E")
st.sidebar.markdown("### 📊 Model Metrics")
st.sidebar.markdown("""
| Model | R² |
|-------|-----|
| Moisture (GB) | 0.918 |
| Solar (MLP) | 0.999 |
| Crop (RF) | 0.977 |
| Demand | Sys.Dyn |
""")
st.sidebar.caption("Phase 2 Capstone | NASA POWER + ICRISAT + Census + CGWB")
