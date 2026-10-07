import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="AUTOSAR & Telemetry Intelligence Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BACKEND_URL = "http://127.0.0.1:8000"

# --- Backend Health Check ---
def check_backend():
    try:
        res = requests.get(f"{BACKEND_URL}/", timeout=2)
        return True
    except Exception:
        return False

backend_online = check_backend()

# --- Sidebar Controls ---
with st.sidebar:
    st.title("⚡ Control Center")
    st.markdown("---")
    
    # System Status Indicator
    if backend_online:
        st.success("🟢 Backend Online")
    else:
        st.error("🔴 Backend Offline")
    
    st.markdown("---")
    st.subheader("1. Ingestion Engine")
    uploaded_file = st.file_uploader("Upload AUTOSAR PDF or Telemetry CSV", type=["pdf", "csv"])
    
    process_btn = st.button(" Process & Embed Document", use_container_width=True)
    
    if process_btn:
        if uploaded_file is not None:
            with st.spinner("Processing document into vector database..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{BACKEND_URL}/ingest", files=files)
                    if response.status_code == 200:
                        res_data = response.json()
                        st.success(f"Ingested **{res_data.get('filename')}** ({res_data.get('chunks_processed')} chunks vectorized).")
                    else:
                        st.error(f"Ingestion Error: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"Connection Failed: {str(e)}")
        else:
            st.warning("Please upload a `.pdf` or `.csv` file first.")

# --- Main Workspace ---
st.title("⚡ AI Assistant Hub")
st.caption("Multimodal RAG Platform for Architectural Specifications & Vehicle Telemetry Analysis")

# Metric Summary Row
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Vector Store", value="ChromaDB")
with col2:
    st.metric(label="Embeddings Engine", value="bge-small-en-v1.5")
with col3:
    st.metric(label="Local LLM Core", value="Llama 3.2")

st.markdown("---")

# Tabbed Layout
tab1, tab2 = st.tabs(["💬 Query & Intelligence", "ℹ️ System Info"])

with tab1:
    st.subheader("Ask Knowledge Base")
    user_query = st.text_input(
        "Enter your question regarding components, interfaces, or telemetry metrics:",
        placeholder="e.g., What is the maximum speed recorded for CAR_001?"
    )
    
    query_btn = st.button(" Run Intelligence Analysis", type="primary")
    
    if query_btn:
        if user_query.strip():
            with st.spinner("Analyzing context & generating response..."):
                try:
                    payload = {"query": user_query}
                    res = requests.post(f"{BACKEND_URL}/query", json=payload)
                    
                    if res.status_code == 200:
                        data = res.json()
                        st.subheader("💡 Analysis Finding")
                        st.info(data.get("answer", "No response generated."))
                        
                        sources = data.get("sources", [])
                        if sources:
                            with st.expander(f"📚 Grounded Evidence & Citations ({len(sources)} Sources)"):
                                for idx, src in enumerate(sources, start=1):
                                    st.markdown(f"**Source Chunk #{idx}**")
                                    st.code(src.get("content"), language="text")
                                    if "metadata" in src:
                                        st.caption(f"Metadata: {src['metadata']}")
                                    st.markdown("---")
                    else:
                        st.error("Error retrieving analysis from backend.")
                except Exception as e:
                    st.error(f"Failed to communicate with backend server: {str(e)}")
        else:
            st.warning("Please enter a query before clicking analyze.")

with tab2:
    st.markdown("""
    ### System Capabilities
    * **AUTOSAR Architecture Ingestion:** Upload official `.pdf` specs to query software components, interfaces, ports, and execution managers.
    * **CSV Telemetry Analysis:** Process tabular logs (`.csv`) to track speed, engine metrics, RPM progression, or statistical data.
    * **Privacy First:** Entirely local pipeline running Ollama (Llama 3.2), ChromaDB, and BGE embeddings without external API reliance.
    """)