"""
RAG Chatbot for Digital Twin - Kalyandurg Village
Uses Gemini embeddings + NumPy cosine similarity (NO FAISS needed)

SETUP:
    pip install google-generativeai numpy
"""

import os
import json
import numpy as np
import time

KNOWLEDGE_BASE = [
    {"title": "ICAR Groundnut Cultivation - Sowing", "content": "For Anantapur district, ICAR recommends groundnut varieties TMV-2, K-6, ICGS-76, and TAG-24. Sowing window is June 15 to July 15, after receiving 50mm cumulative rainfall. Seed rate is 100 kg/ha for spreading types and 120 kg/ha for bunch types. Recommended spacing is 30cm between rows and 10cm between plants. Early sowing after first good rain gives 15-20% higher yield than late sowing."},
    {"title": "ICAR Groundnut - Water Management", "content": "Critical irrigation stages for groundnut are Pegging stage (35-40 days after sowing) and Pod formation stage (60-70 days after sowing). Protective irrigation at pegging stage alone increases yield by 25-30%. Total water requirement is 450-500mm for the full crop cycle. Soil moisture should not fall below 25% during critical growth stages. In red sandy loam soils of Anantapur, moisture depletes faster due to low water holding capacity. Drip irrigation at 0.6 CPE saves 40% water compared to flood irrigation."},
    {"title": "PM-KUSUM Scheme - Component B", "content": "PM-KUSUM Component B provides standalone solar agricultural pumps up to 7.5 HP. Subsidy: 30% CFA from Central + 30% from State + 40% by farmer (bank loan available). Eligible: All farmers with borewells or open wells. A 5 HP solar pump irrigates 2-3 acres and saves Rs 50,000-70,000 per year in electricity/diesel costs. Ideal for Anantapur where grid power supply is erratic."},
    {"title": "Jal Jeevan Mission - Rural Water Supply", "content": "Jal Jeevan Mission targets 55 LPCD potable water to every rural household via tap connections. AP has ~65% rural households connected as of 2024. Emphasizes source sustainability through groundwater recharge, rainwater harvesting, and grey water management. For Anantapur, JJM mandates water budgeting at village level before approving new connections."},
    {"title": "Anantapur District Climate Profile", "content": "Anantapur is the driest district in AP with 544mm average annual rainfall. 61% from SW monsoon (June-September), 25% from NE monsoon (October-December). Droughts occur 1 in every 3-4 years. Temperature: 20C in December to 42C in May. 250-280 clear sky days per year. Solar irradiance: 5.5 kWh/m2/day - one of the best in India for solar energy."},
    {"title": "Anantapur Groundwater Status", "content": "Several mandals classified as Over-Exploited or Critical by CGWB. Kalyandurg mandal: groundwater at 7-10m depth, seasonal variation of 2-4m (deepest May at 10-12m, shallowest November at 7-8m). District experienced 3.89m depletion in 2023 due to drought. Real data from AP WRIMS shows Kalyandurg at 7.05m as of March 2026. Farmers advised to adopt micro-irrigation and build rainwater harvesting structures."},
    {"title": "PMFBY Crop Insurance for Anantapur", "content": "Pradhan Mantri Fasal Bima Yojana: Kharif crops at 2% premium, Rabi at 1.5%. Notified crops: Groundnut, Sunflower, Castor, Chickpea, Red gram, Sorghum. Covers drought, flood, hailstorm, pest/disease. Claims within 72 hours via app or agriculture officer. Sum insured for groundnut: ~Rs 35,000/hectare for 2024-25."},
    {"title": "Solar Energy Potential in Anantapur", "content": "Global Horizontal Irradiance: 5.5 kWh/m2/day. CUF: 18-22%. 1kW system generates 4-5 kWh/day. 5kW system powers a 3HP pump for 6-8 hours daily. Efficiency drops 0.4% per degree above 25C. Annual degradation 0.5-0.7%. 5kW system costs Rs 2.5-3 lakhs before subsidy."},
]


class AgriRAG:
    def __init__(self, api_key):
        self.api_key = api_key
        self.chunks = KNOWLEDGE_BASE
        self.embeddings = None
        self.genai = None
        self.gen_model = None
        self._setup_gemini()
    
    def _setup_gemini(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
            self.gen_model = genai.GenerativeModel("gemini-2.5-flash")
            print("Gemini API connected")
        except Exception as e:
            print(f"Gemini setup failed: {e}")
    
    def _embed_text(self, text, task="retrieval_document"):
        result = self.genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type=task
        )
        return np.array(result['embedding'], dtype=np.float32)
    
    def initialize(self):
        if not self.genai:
            return False
        print("Building RAG knowledge base...")
        emb_list = []
        for i, chunk in enumerate(self.chunks):
            text = f"{chunk['title']}: {chunk['content']}"
            emb = self._embed_text(text)
            emb_list.append(emb)
            print(f"  Embedded {i+1}/{len(self.chunks)}: {chunk['title']}")
            if i < len(self.chunks) - 1:
                time.sleep(8)  # 4 sec delay = ~15 calls per minute
        self.embeddings = np.vstack(emb_list)
        print(f"Knowledge base ready: {len(self.chunks)} chunks, {self.embeddings.shape[1]}D")
        return True
    
    def retrieve(self, query, k=5):
        if self.embeddings is None:
            return []
        q_emb = self._embed_text(query, task="retrieval_query")
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        e_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10)
        scores = np.dot(e_norm, q_norm)
        top_idx = np.argsort(scores)[::-1][:k]
        return [{"title": self.chunks[i]["title"], "content": self.chunks[i]["content"], "score": float(scores[i])} for i in top_idx]
    
    def query(self, question, simulation_context=None):
        if not self.genai or self.embeddings is None:
            return "RAG not ready. Check API key and call initialize()."
        
        retrieved = self.retrieve(question, k=5)
        rag_context = "\n\n".join([f"**{r['title']}** (relevance: {r['score']:.2f}):\n{r['content']}" for r in retrieved])
        
        sim_info = ""
        if simulation_context:
            sim_info = "\n".join([f"- {k}: {v}" for k, v in simulation_context.items()])
            sim_info = f"\n## Current Simulation State:\n{sim_info}\n"
        
        prompt = f"""You are an expert agricultural advisor for Kalyandurg village, Anantapur district, AP, India. You have ICAR guidelines, government scheme documents, and real-time Digital Twin simulation data.
{sim_info}
## Knowledge Base:
{rag_context}

## Question: {question}

Answer using simulation data + knowledge base. Be specific to Anantapur conditions. Reference schemes with details. Give actionable advice with numbers. Use bullet points for actions."""
        
        try:
            return self.gen_model.generate_content(prompt).text
        except Exception as e:
            return f"Error: {e}"
    
    def get_sources(self, question):
        return [(r["title"], r["score"]) for r in self.retrieve(question, k=3)]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rag_system.py YOUR_GEMINI_API_KEY")
        sys.exit(1)
    rag = AgriRAG(sys.argv[1])
    if rag.initialize():
        print("\nTest query: Should I irrigate today?")
        print(rag.query("Should I irrigate my groundnut today?", {"soil_moisture": 28.5, "groundwater_depth": 8.99, "crop": "Groundnut"}))
