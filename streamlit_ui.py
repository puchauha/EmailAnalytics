# %%
# streamlit_outage_ui.py

import os
import streamlit as st
from dotenv import load_dotenv
import duckdb
from outage_analyzer import build_outage_agent_graph  # ✅ your tested LangGraph builder

# -------------------------------------------------------------------
# 1️⃣ Setup Environment
# -------------------------------------------------------------------
load_dotenv()

# Explicitly set key for LangChain/OpenAI clients
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not found. Please set it in .env or Streamlit secrets.")
    st.stop()

# -------------------------------------------------------------------
# 2️⃣ Initialize DuckDB
# -------------------------------------------------------------------
@st.cache_resource
def get_connection():
    con = duckdb.connect("outages.duckdb")
    return con

con = get_connection()
st.sidebar.success("💾 Connected to outages.duckdb")

# -------------------------------------------------------------------
# 3️⃣ Initialize LangGraph App
# -------------------------------------------------------------------
@st.cache_resource
def get_langgraph_app():
    return build_outage_agent_graph()

app = get_langgraph_app()

# -------------------------------------------------------------------
# 4️⃣ Streamlit UI
# -------------------------------------------------------------------
st.title("🔌 Outage Analytics Agent")
st.caption("Powered by LangGraph + GPT reasoning")

# User query input
user_query = st.text_area("Ask your question about outages:", "Show total outages per partner")

if st.button("Run Analysis 🚀"):
    if not user_query.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("🤖 Thinking... running LangGraph agent..."):
        try:
            response = app.invoke({"user_query": user_query, "db_con": con})
        except Exception as e:
            st.error(f"❌ Error during agent execution: {e}")
            st.stop()

    # -------------------------------------------------------------------
    # 5️⃣ Display Results
    # -------------------------------------------------------------------
    st.subheader("🧠 Agent Response")

    final_answer = response.get("final_answer") if isinstance(response, dict) else response
    if not final_answer:
        st.warning("⚠️ No final answer returned from the agent.")
    else:
        st.markdown(final_answer)

    # Optional: Display intermediate outputs (SQL, chart URIs, etc.)
    if isinstance(response, dict):
        if response.get("sql_query"):
            st.markdown("### 🧮 Generated SQL")
            st.code(response["sql_query"], language="sql")

        if response.get("chart_uri"):
            st.markdown("### 📊 Chart Visualization")
            st.image(response["chart_uri"])

        if response.get("summary"):
            st.markdown("### 📝 Summary")
            st.write(response["summary"])

# -------------------------------------------------------------------
# 6️⃣ Optional: Diagnostics Sidebar
# -------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Diagnostics")
st.sidebar.text(f"API key loaded: {bool(os.getenv('OPENAI_API_KEY'))}")



