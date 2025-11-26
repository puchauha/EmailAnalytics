# %%
# app.py
# ============================================================
# Outage Analytics Assistant
# - Data source: outages.xlsx (Pandas DataFrame)
# - LLM: gpt-4o-mini via langchain_openai
# - Orchestration: LangGraph StateGraph
# - UI: Streamlit chat
# ============================================================

from __future__ import annotations

from typing import Any, TypedDict
import base64
import io
import traceback

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import streamlit as st

# ------------------------------------------------------------
# 1. Environment + LLM + Data
# ------------------------------------------------------------
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

OUTAGES_EXCEL = "outages.xlsx"
# Global DataFrame used by the agent
df_outages = pd.read_excel(OUTAGES_EXCEL)


# ------------------------------------------------------------
# 2. LangGraph State definition
# ------------------------------------------------------------
class AgentState(TypedDict, total=False):
    user_query: str
    pandas_code: str
    result: Any


# ------------------------------------------------------------
# 3. Node: Generate Pandas code from user query
# ------------------------------------------------------------
def generate_pandas_code(state: AgentState) -> dict:
    """Node 1: LLM generates Pandas code from the user's query."""
    user_query = state.get("user_query") or state.get("__input__", {}).get("user_query")
    if not user_query:
        raise KeyError("Missing 'user_query' in state")

    prompt = f"""
    You are a Python data analysis assistant.

    You have a Pandas DataFrame named df with these columns:
    - partner_name (str)
    - outage_type (str)
    - issue_details (str)
    - current_status (str)
    - business_impact (str)
    - manual_processing (str or bool)
    - outage_start_time (datetime or str)
    - outage_end_time (datetime or str)
    - duration_hours (float)

    User request:
    \"\"\"{user_query}\"\"\"

    GENERAL RULES:
    - Use ONLY Python and Pandas (no SQL).
    - Assume df is already defined and contains all data.
    - You may use:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import io
        import base64
    - Always assign the final answer to a variable called `result`.
    - Never wrap your code in markdown or backticks. Return RAW Python code only.

    DATE HANDLING RULE:
    - Before filtering by date, always convert date/time fields to datetime using:
        df['outage_start_time'] = pd.to_datetime(df['outage_start_time'], errors='coerce')
        if 'email_date' in df.columns:
            df['email_date'] = pd.to_datetime(df['email_date'], errors='coerce')

    EXECUTIVE SUMMARY MODE:
    - If the user is asking for an "executive summary", "leadership summary",
    or "summary for leadership":
        - Filter the data by time if the user specifies a period (e.g. "last 3 months",
        "last 5 years") using pd.Timestamp.now() and pd.DateOffset.
        - Compute high-level KPIs such as:
            total_outages (int)
            total_downtime_hours (float) = sum of duration_hours
            avg_outage_duration_hours (float)
            outages_per_partner (counts)
            top_3_partners_by_downtime
            unique_business_impacts
        - Build a concise multi-sentence summary string called `summary_text`.
        - Build a compact table DataFrame called `summary_table` with one row per partner
        and columns like:
        ['partner_name', 'outage_count', 'total_downtime_hours', 'avg_duration_hours'].
        - Set:
            result = {{
                "type": "executive_summary",
                "summary_text": summary_text,
                "summary_table": summary_table
            }}
        - Do NOT generate charts in this mode unless the user explicitly asks for a chart.

    CHART MODE (non-executive queries where user explicitly asks for chart/plot/graph):
    - If the user clearly asks for a chart/plot/graph:
        - Use matplotlib: `import matplotlib.pyplot as plt`.
        - Build the chart using df.
        - Save it to a base64 PNG string:
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            result = {{"type": "chart", "image_base64": img_str}}
        - Do NOT call plt.show().

    RENAME RULES (to avoid Pandas errors):
    - Never call Series.rename(columns=...).
    - If you need to rename a Series, either:
        - convert it to a DataFrame with `.to_frame(name="colname")`, or
        - leave the default name.
    - Use `.rename(columns={{...}})` ONLY on DataFrames.

    NON-CHART, NON-EXECUTIVE CASES:
    - If no chart and no executive summary is requested:
        - `result` can be:
            - a DataFrame
            - a Series
            - a list / dict
            - a scalar (number/string).

    -- Always normalize datetimes to tz-naive using:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[col] = df[col].dt.tz_localize(None)


    Remember:
    - Always assign the final answer to a variable called `result`.
    - Return only the Python code implementing the above logic.
    """

    code = llm.invoke(prompt).content.strip()
    return {"pandas_code": code}




# ------------------------------------------------------------
# 5. Node: Wrap execute_pandas_code for LangGraph
# ------------------------------------------------------------
def execute_pandas_node(state: AgentState) -> dict:
    """Node 2: takes the generated code and runs it against df_outages."""
    code = state.get("pandas_code", "")
    if not code:
        return {
            "result": {
                "type": "error",
                "error_message": "No pandas_code found in state.",
            }
        }

    result = execute_pandas_code(df_outages, code)
    return {"result": result}


# ------------------------------------------------------------
# 6. Build LangGraph agent
# ------------------------------------------------------------
def build_outage_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("generate_pandas_code", generate_pandas_code)
    graph.add_node("execute_pandas", execute_pandas_node)

    graph.set_entry_point("generate_pandas_code")
    graph.add_edge("generate_pandas_code", "execute_pandas")
    graph.add_edge("execute_pandas", END)

    return graph.compile()






# %%
import pandas as pd
import matplotlib.pyplot as plt
import io, base64
import traceback

def execute_pandas_code(df: pd.DataFrame, code: str):
    """
    Executes LLM-generated Pandas code safely and returns structured results.
    Handles:
      - Executive summaries (summary_text + summary_table)
      - DataFrames
      - Charts (matplotlib saved to base64)
    Adds:
      - Auto datetime + timezone normalization
      - Better error handling + output detection
    """

    # Local env for execution
    local_env = {
    "df": df.copy(),
    "pd": pd,
    "plt": plt,
    "io": io,
    "base64": base64
}


    # Safe builtins — prevents arbitrary code execution
    exec_globals = {"__builtins__": {"len": len, "range": range, "min": min, "max": max, "sum": sum}}

    # 🔧 Auto-fix: datetime conversion + tz-naive normalization
    for col in df.columns:
        lower = col.lower()
        if "date" in lower or "time" in lower:
            try:
                local_env["df"][col] = pd.to_datetime(local_env["df"][col], errors="coerce")
                # Drop timezone awareness always (prevents comparison errors)
                if hasattr(local_env["df"][col].dt, "tz_localize"):
                    try:
                        local_env["df"][col] = local_env["df"][col].dt.tz_localize(None)
                    except TypeError:
                        # If timezone exists before dropping
                        try:
                            local_env["df"][col] = local_env["df"][col].dt.tz_convert(None)
                        except Exception:
                            pass
            except Exception:
                pass

    print("\n📌 Running Pandas code:\n", code)

    # 🚀 Execute LLM code
    try:
        exec(code, exec_globals, local_env)
    except Exception as e:
        return {
            "type": "error",
            "error_message": str(e),
            "trace": traceback.format_exc(),
            "failed_code": code
        }

    # 🎯 Extract final result smartly
    # Executive summary mode
    if "summary_text" in local_env:
        result = {"type": "executive_summary"}
        result["summary_text"] = local_env.get("summary_text", "")

        if "summary_table" in local_env:
            summary_df = local_env["summary_table"]
            result["summary_table"] = summary_df.to_dict(orient="records")
            result["columns"] = list(summary_df.columns)

        return result

    # Chart output
    figs = list(map(plt.figure, plt.get_fignums()))
    if figs:
        buf = io.BytesIO()
        figs[-1].savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close("all")

        return {"type": "chart", "image_base64": img_str}

    # DataFrame output
    if isinstance(local_env.get("result"), pd.DataFrame):
        df_res = local_env["result"]
        return {
            "type": "table",
            "data": df_res.to_dict(orient="records"),
            "columns": df_res.columns.tolist()
        }

    # Generic python output
    if "result" in local_env:
        return {"type": "value", "value": local_env["result"]}

    return {"type": "unknown", "message": "Code executed but produced no recognized output"}


# %%
# ------------------------------------------------------------
# 7. Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Outage Analytics Assistant", layout="wide")
st.title("🔌 Outage Analytics Assistant (Pandas + LangGraph)")


@st.cache_resource
def load_agent():
    return build_outage_agent_graph()


agent = load_agent()

# Sidebar: Data info
st.sidebar.header("Dataset Info")
st.sidebar.write(f"Rows: {len(df_outages)}")
if st.sidebar.checkbox("Show raw data"):
    st.sidebar.dataframe(df_outages, use_container_width=True)

st.markdown("Ask questions like:")
st.markdown("- *Show total outages per partner*")
st.markdown("- *List unique business impacts for the last 5 years*")
st.markdown("- *Show average outage duration per partner as a bar chart*")
st.markdown("- *Give an executive summary for leadership for the last 3 months*")

if "history" not in st.session_state:
    st.session_state.history = []


def render_assistant_content(content: Any):
    """Helper to render assistant message content based on its type."""
    # Chart (dict from LLM)
    if isinstance(content, dict) and content.get("type") == "chart":
        img_b64 = content.get("image_base64")
        if img_b64:
            img_bytes = base64.b64decode(img_b64)
            st.image(img_bytes, caption="Generated chart")
        else:
            st.write(content)
        return

    # Executive summary
    if isinstance(content, dict) and content.get("type") == "executive_summary":
        st.subheader("📌 Executive Summary")
        st.write(content.get("summary_text", ""))

        if "summary_table" in content:
            st.subheader("📊 Summary Table")
            st.dataframe(pd.DataFrame(content["summary_table"]), use_container_width=True)
        return

    # Table (generic)
    if isinstance(content, dict) and content.get("type") == "table" and "data" in content:
        st.dataframe(pd.DataFrame(content["data"]), use_container_width=True)
        return

    # DataFrame directly
    if isinstance(content, pd.DataFrame):
        st.dataframe(content, use_container_width=True)
        return

    # Fallback: just print
    st.write(content)


# Show previous chat
for role, content in st.session_state.history:
    with st.chat_message(role):
        if role == "assistant":
            render_assistant_content(content)
        else:
            st.write(content)

# Chat input
query = st.chat_input("Ask about outages…")

if query:
    # User message
    st.session_state.history.append(("user", query))
    with st.chat_message("user"):
        st.write(query)

    # Agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking with Pandas…"):
            state = {"user_query": query}
            out = agent.invoke(state)

        result = out.get("result")
        code = out.get("pandas_code")

        render_assistant_content(result)
        st.session_state.history.append(("assistant", result))

        # Optional: show generated Pandas code for debugging
        with st.expander("Show generated Pandas code"):
            st.code(code or "No code generated.", language="python")



# %%
# ------------------------------------------------------------
# 8. CLI test (if run as pure script, not via streamlit)
# ------------------------------------------------------------
if __name__ == "__main__":
    print("💾 Loaded outages.xlsx with", len(df_outages), "rows")
    app = build_outage_agent_graph()
    test_state = {"user_query": "creat"}
    out = app.invoke(test_state)
    print("=== Generated code ===")
    print(out.get("pandas_code"))
    print("\n=== Result ===")
    print(out.get("result"))


