# %%
# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

# %%
from __future__ import annotations

from typing import Any, TypedDict
import base64
import io
import traceback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import streamlit as st
load_dotenv()

# %%
# ------------------------------------------------------------
# Environment + LLM + Data
# ------------------------------------------------------------


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0,api_key=st.secrets["OPENAI_API_KEY"])

OUTAGES_EXCEL = "outages.xlsx"
# Global DataFrame used by the agent
df_outages = pd.read_excel(OUTAGES_EXCEL)

# %%
# ------------------------------------------------------------
# Prompt definition
# ------------------------------------------------------------
prompt_summary = """
You are an expert Python data analyst.

You write Pandas code to work with a DataFrame named `df` having these columns:
- partner_name (str)
- outage_type (str)
- issue_details (str)
- current_status (str)
- business_impact (str)
- manual_processing (str/bool)
- outage_start_time (datetime/str)
- outage_end_time (datetime/str)
- duration_hours (float)
"""

# %%
# ------------------------------------------------------------
# Prompt definition
# ------------------------------------------------------------

execution_rules = """

IMPORTANT EXECUTION RULES:
- Code MUST be plain Python (NO markdown, NO comments).
- Do NOT import pandas or create new DataFrames — `df` is already provided.
- Convert all datetime fields using:
    df[col] = pd.to_datetime(df[col], errors='coerce').dt.tz_localize(None)
- BEFORE any filtering by dates or plotting.
- ALWAYS define a final variable named `result`. This is what will be returned.
- DO NOT add source file and parsed date details

DO NOT write import statements — they will cause execution failure.
You must only use the following already-available objects:
- df  : Pandas DataFrame loaded with outage data
- pd  : pandas module
- plt : matplotlib.pyplot
- io  : io module for BytesIO
- base64 : for encoding charts
- np  : numpy module

If you need a chart:
- Use `plt.figure()` before plotting
- Save using:
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    result = {"type": "chart", "image_base64": img_str}
- NEVER call plt.show()

CHART STYLE RULES:
- Chart size must be professional and compact:
    plt.figure(figsize=(6, 4))
- Avoid full-screen or oversized charts.
- Use clean white background (default).
- Titles should be concise and readable.
- Avoid legends if labels are already visible (e.g. pie slices or x-axis labels).
- Rotate x-axis labels only if overlapping: plt.xticks(rotation=45)
- No alpha transparency effects or neon colors.
- Prefer:
    - Bar chart → plt.bar()
    - Line chart → plt.plot()
    - Pie chart → plt.pie() only if < 8 slices
- Apply padding for neat layout:
    plt.tight_layout()


If you produce a single number or string, return:
    result = {"type": "text_value", "value": <python_value>}


AGGREGATION RULES FOR PARTNER-LEVEL OUTPUT:
- If the user asks for partner-level outage summaries:
    - Ensure exactly one row per partner.
    - Include only fields that can be aggregated per partner.
    - Aggregate numeric columns like:
        - outage_count → count()
        - total_downtime_hours → sum(duration_hours)
        - avg_duration_hours → mean(duration_hours)
    - For business_impact & issue_details:
        - Create separate lists using .unique().tolist()
        - Name them: unique_business_impacts, unique_issues
        
    - Do not include raw text columns that cannot be aggregated (like issue_details as a long string)

    SCALAR RESULT RULE:
    - If the user requests a single answer such as:
        - highest / lowest / maximum / minimum outages or downtime
        - "Which partner has the most outages?"
        - "Show the average downtime overall"
        - "How many unique partners had outages?"
    - DO NOT create a DataFrame or table.
    - Instead compute the metric and return a dict:

    result = {
        "type": "text_value",
        "text": "MegaTrans Global has the maximum outages (14)."
    }

    - Only use DataFrames if user requests multiple rows of results.


HEATMAP RULE (STRICT AND NON-NEGOTIABLE):

- If the user mentions "heatmap", the output MUST be a heatmap.
- DO NOT generate bar charts, pie charts, line charts, or tables.
- ONLY generate a heatmap using matplotlib + pivot_table.

Fuzzy Partner Matching — STRICT RULE:
- NEVER perform exact equality matching like df[df["partner_name"] == "x"].
- ALWAYS use `.str.contains(term, case=False, na=False)` for matching.
- ALWAYS use `~ .str.contains()` for exclusions.

This overrides ALL other filtering rules.

PARTNER FILTERING RULE FOR HEATMAP:
- Use fuzzy partial matching for includes:
      df[df["partner_name"].str.contains(<term>, case=False, na=False)]
- Use fuzzy partial matching for exclusions:
      df[~df["partner_name"].str.contains(<term>, case=False, na=False)]

This rule overrides all other chart generation logic.


PARTNER NAME MATCHING RULE:
- Partner filters must use partial, case-insensitive matching:
  
      df[df["partner_name"].str.contains(<pattern>, case=False, na=False)]

- Never use equality like:
      df[df["partner_name"] == "ExpressLine"]

- For exclusions:

      df[~df["partner_name"].str.contains(<pattern>, case=False, na=False)]

- If the query contains any mention of a partner name (full or partial), extract it and apply fuzzy matching.

- This ensures user typos, short forms, and partial names still return valid results.

        
    
    OUTPUT RULES:
    If user asks for counts, unique values, lists → assign to `result` (list/dict/DataFrame).
    If user asks a chart:
    - Import: import matplotlib.pyplot as plt, import io, import base64, import numpy as np
    - Create figure: plt.figure(figsize=(10,6))
    - Save chart as base64:
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        result = {"type": "chart", "image_base64": img_str}
    - Do NOT use plt.show()

    DECISION LOGIC (VERY IMPORTANT):
    - If the user asks for a CHART,TRENDS, BAR, PIE, PARETO,Image → return type: "chart"
    - If the user asks for a TABLE → return type: "table"
    - If the request is about BUSINESS IMPACT/TRENDS/AGGREGATE INSIGHTS → return type: "executive_summary"
    - If unclear → favor "table"

    Never let Streamlit decide presentation — YOU decide based on query.



    If grouping by partner or issue:
        - Use .groupby([...], dropna=True)

    For renaming:
        - DataFrame.rename(columns={{...}}) — OK
        - NEVER call Series.rename(columns=...)

   EXECUTIVE SUMMARY MODE (only if user *explicitly* requests it):
- Activated only if user mentions one of:
  ["executive summary", "leadership summary", "summary for leadership", "summary report"]
- Produce both:
   * summary_text (3–4 business insights in English)
   * summary_table (per-partner KPI breakdown)
- Avoid including non-numeric fields in aggregated tables
- Include system chart JSON only if user explicitly mentions chart


    FINAL REQUIREMENT:
    - Your FINAL LINE must be the assignment: result = ...
    - NEVER print or display charts in code.
    - Return ONLY the code. No markdown. No explanation.

    """

# %%
def is_valid_query(user_query: str) -> bool:
    """LLM classifies whether the query is meaningful or garbage."""
    check_prompt = f"""
You are a query classifier for an outage analytics assistant.
User query: "{user_query}"

Decide ONLY:
- "valid" → if query relates to outages, trends, charts, partners, issues, downtime, etc.
- "invalid" → if it is nonsense, random words, or unprocessable.

Answer only one word: valid or invalid.
"""
    print("📝 Validating user query...")
    print("Prompt:", check_prompt)
    result = llm.invoke(check_prompt).content.strip().lower()
    return result == "valid"


# %%
# ------------------------------------------------------------
# Helper imports for execute_pandas_code
# ------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import traceback


def execute_pandas_code(df: pd.DataFrame, code: str):
    """
    Executes LLM-generated Pandas code safely and returns a structured result.

    The LLM is expected to use a DataFrame named `df`.

    Possible returned structures:
      - {"type": "table", "rows": [...], "columns": [...]}
      - {"type": "chart", "image_base64": "..."}
      - {"type": "text", "value": "..."}
      - {"type": "executive_summary", "summary_text": str, "rows": [...], "columns": [...]}
      - {"type": "error", "error_message": str, "trace": str, "failed_code": str}
      - {"type": "unknown", "message": str}

    Notes:
      - Strips any `import ...` lines the LLM might generate.
      - Normalizes all date/time columns to tz-naive datetimes.
    """

    # ---------- 1. Local sandbox environment ----------
    local_env = {
        "df": df.copy(),      # work on a copy so original df_outages is safe
        "pd": pd,
        "plt": plt,
        "np": np,
        "io": io,
        "base64": base64,
    }

    # Very restricted builtins to avoid dangerous operations
    exec_globals = {
        "__builtins__": {
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "float": float,  # <-- Added support for float()
            "int": int,      # <-- Optional but recommended
            "str": str,
        }
    }

    # ---------- 2. Auto datetime + timezone normalization ----------
    for col in df.columns:
        lower = col.lower()
        if "date" in lower or "time" in lower:
            try:
                # Convert to datetime where possible
                local_env["df"][col] = pd.to_datetime(
                    local_env["df"][col], errors="coerce"
                )
                # Drop timezone info if present (tz-aware vs tz-naive issues)
                if hasattr(local_env["df"][col].dt, "tz_localize"):
                    try:
                        local_env["df"][col] = local_env["df"][col].dt.tz_localize(None)
                    except TypeError:
                        # In some cases tz_convert is needed first
                        try:
                            local_env["df"][col] = (
                                local_env["df"][col].dt.tz_convert(None)
                            )
                        except Exception:
                            pass
            except Exception:
                # If conversion fails, just leave the column as is
                pass

    # ---------- 3. Strip any leading import lines from LLM code ----------
    cleaned_lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            # Skip imports (we already provided pd, plt, io, base64, np)
            continue
        cleaned_lines.append(line)
    cleaned_code = "\n".join(cleaned_lines)

    print("\n📌 Running Pandas code:######\n", cleaned_code)
    

    # ---------- 4. Execute the LLM-generated code ----------
    try:
        exec(cleaned_code, exec_globals, local_env)
    except Exception as e:
        print("❌ Pandas execution error:", e)
        print(traceback.format_exc())
        # Return user-friendly error for UI
        return {
        "type": "friendly_error",
        "message": (
            "I apologize — I couldn't understand your query. "
            "Please rephrase, break, or simplify it for better results."
            )
          }

    # ---------- 5. Inspect outputs in priority order ----------

    # 5.1 If LLM explicitly set a dict `result` with "type", trust it
    result = local_env.get("result")
    if isinstance(result, dict) and "type" in result:
        # For table-like dicts the LLM might have used keys like 'data'
        # We normalize to rows/columns if it's a DataFrame inside
        if isinstance(result.get("data"), pd.DataFrame):
            df_res = result["data"]
            return {
                "type": result.get("type", "table"),
                "rows": df_res.to_dict(orient="records"),
                "columns": list(df_res.columns),
            }
        return result

    # 5.2 Executive summary via summary_text + summary_table
    if "summary_text" in local_env and "summary_table" in local_env:
        summary_table = local_env["summary_table"]
        if isinstance(summary_table, pd.DataFrame):
            return {
                "type": "executive_summary",
                "summary_text": str(local_env["summary_text"]),
                "rows": summary_table.to_dict(orient="records"),
                "columns": list(summary_table.columns),
            }

    # 5.3 Chart from matplotlib (if any figure was created)
    figs = list(map(plt.figure, plt.get_fignums()))
    if figs:
        buf = io.BytesIO()
        # Use the last created figure
        figs[-1].savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close("all")
        return {"type": "chart", "image_base64": img_str}

    # 5.4 DataFrame returned as result
    if isinstance(result, pd.DataFrame):
        return {
            "type": "table",
            "rows": result.to_dict(orient="records"),
            "columns": list(result.columns),
        }

    # 5.5 List / Series / ndarray → represent as a 1-column table
    if isinstance(result, (list, np.ndarray, pd.Series)):
        df_res = pd.DataFrame({"value": list(result)})
        return {
            "type": "table",
            "rows": df_res.to_dict(orient="records"),
            "columns": ["value"],
        }

    # 5.6 Scalar (int/float/str/bool) → text
    if isinstance(result, (int, float, str, bool)):
        return {"type": "text", "value": str(result)}

    # 5.7 If LLM modified df in-place and didn't set result
    if "df" in local_env and isinstance(local_env["df"], pd.DataFrame):
        df_mod = local_env["df"]
        return {
            "type": "table",
            "rows": df_mod.to_dict(orient="records"),
            "columns": list(df_mod.columns),
        }

    # 5.8 Fallback: nothing recognizable
    return {
        "type": "unknown",
        "message": "Code executed but produced no recognized output (no result, chart, or DataFrame).",
    }


# %%
# ------------------------------------------------------------
# LangGraph State definition
# ------------------------------------------------------------
class AgentState(TypedDict, total=False):
    user_query: str
    pandas_code: str
    result: Any

# %%
# ------------------------------------------------------------
# Node: Generate Pandas code from user query
# ------------------------------------------------------------
def generate_pandas_code(state: AgentState) -> dict:
    """Node 1: LLM generates Pandas code from the user's query."""
    user_query = state.get("user_query") or state.get("__input__", {}).get("user_query")
    if not user_query:
        raise KeyError("Missing 'user_query' in state")

    prompt = prompt_summary +  " User request : " +user_query + execution_rules 
    

    code = llm.invoke(prompt).content.strip()
    return {"pandas_code": code}


# %%
# ------------------------------------------------------------
# Node: Wrap execute_pandas_code for LangGraph
# -----------------------------------------------------------
def execute_pandas_node(state: AgentState) -> dict:
    """Node 2: takes the generated code and runs it against df_outages."""
    
    user_query = state.get("user_query", "")
    code = state.get("pandas_code", "")
    
    # 🔍 1. Validate query BEFORE execution
    if not is_valid_query(user_query):
        return {
            "result": {
                "type": "friendly_error",
                "message": (
                    "I apologize — I couldn't understand your query. "
                    "Please rephrase, break, or simplify it."
                )
            }
        }

    # ❗ Continue normally if valid
    if not code:
        return {
            "result": {
                "type": "error",
                "error_message": "No pandas_code found in state.",
            }
        }

    result = execute_pandas_code(df_outages, code)
    return {"result": result}


# %%
def generate_executive_summary(state: AgentState):
    result = state.get("result")
    user_query = state.get("user_query")
    
    if not result or (
        result.get("type") not in ["table", "chart", "text_value"]
    ):
        return {"summary": None}

    # If chart already exists, include caption + insights only
    if result.get("type") == "chart":
        summary_text = result.get("summary_text", "Chart generated for key outage insights.")
        return {
            "summary": summary_text,
            "chart_uri": result.get("image_base64")
        }

    # Standard summary logic follows...


# %%
# ------------------------------------------------------------
# Build LangGraph agent #####&&&
# ------------------------------------------------------------
def build_outage_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("generate_pandas_code", generate_pandas_code)
    graph.add_node("execute_pandas", execute_pandas_node)
    graph.add_node("generate_executive_summary", generate_executive_summary)

    graph.set_entry_point("generate_pandas_code")
    graph.add_edge("generate_pandas_code", "execute_pandas")
    graph.add_edge("execute_pandas", "generate_executive_summary")
    graph.add_edge("generate_executive_summary", END)

    return graph.compile()



# %%
import streamlit as st
import base64
import pandas as pd

st.set_page_config(page_title="📊 Partner Outage Analytics Assistant", layout="wide")
st.title("📊 Partner Outage Analytics Assistant")

st.markdown("""
### 📘 How to Use the Partner Outage Analytics Assistant

Ask questions in **plain English** — the system automatically understands, analyzes, 
and generates results using **Pandas + AI reasoning**.

You can request:
- 📊 **Charts** — bar, line, pie, heatmap, trends  
- 📋 **Tables** — outage counts, partner summaries, unique issues, etc.  
- 🧮 **KPIs** — totals, averages, maximums, comparisons  
- 📝 **Executive summaries** *(only if explicitly mentioned)*  

The assistant automatically:
- Converts dates & data types  
- Aggregates results correctly  
- Picks the right visual or table format  
- Avoids long text fields in aggregated tables  
- Handles partial partner name matches  
- Generates compact, professional charts  

---

### 🧠 Example Queries

#### 📌 Outage Insights  
- *“Show total outages per year”*  
- *“Outage trend for last 6 months”*  
- *“List all unique issues for 2025”*  
- *“Top 3 partners by downtime”*  

#### 📌 Partner-Specific  
- *“Outage summary for MegaTrans Global for last 3 years”*  
- *“Issues and business impacts for ExpressLine”*  
- *“Total downtime for BlueRoute in 2024”*  

#### 📌 Charts  
- *“Create a bar graph of outages per partner”*  
- *“Heatmap of outages by partner and issue type”*  
- *“Pie chart of outage types for 2025”*  

#### 📌 Advanced Filtering  
- *“Show outages excluding ExpressLine”*  
- *“Heatmap of outages for all partners except BlueRoute”*  
- *“Trend of critical outages only”*  

#### 📝 Executive Summary (must be explicit)  
- *“Give executive summary for partner outages in 2024”*  
- *“Executive summary of last 3 months outages with insights”*  

---

### ✔️ Tip  
If you do NOT specify “executive summary,” the system returns  
a **table** or **chart** based on your query.

""")


@st.cache_data
def load_df():
    return pd.read_excel("outages.xlsx")

df_outages = load_df()
agent = build_outage_agent_graph()

#st.sidebar.write(f"Dataset rows: {len(df_outages)}")

query = st.chat_input("Ask a question about outages…")

if "history" not in st.session_state:
    st.session_state.history = []

# Render existing history
for role, content in st.session_state.history:
    with st.chat_message(role):
        if isinstance(content, dict):
            if content.get("type") == "chart":
                st.image(base64.b64decode(content["image_base64"]))

            elif content.get("type") == "table":
                st.dataframe(content["rows"], use_container_width=True)

            elif content.get("type") in ["text", "text_value"]:
                st.write(content.get("text") or content.get("value"))

            elif content.get("type") == "executive_summary":
                st.subheader("📌 Executive Summary")
                st.write(content["summary_text"])
                if "rows" in content:
                    st.dataframe(content["rows"], use_container_width=True)
                if "image_base64" in content:
                    st.image(base64.b64decode(content["image_base64"]))

        else:
            st.write(content)

if query:
    # Show user query bubble
    st.session_state.history.append(("user", query))
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Processing…"):
            out = agent.invoke({"user_query": query})
            result = out.get("result")

        # Normalize and display instantly
        clean_result = result if isinstance(result, dict) else {
            "type": "text",
            "value": str(result)
        }

        # Append cleaned result to chat history
        st.session_state.history.append(("assistant", clean_result))

        # Render output now
        if clean_result["type"] == "table":
            st.dataframe(clean_result["rows"], use_container_width=True)

        elif clean_result["type"] == "chart":
            st.image(base64.b64decode(clean_result["image_base64"]), caption="Generated Chart")

        elif clean_result["type"] in ["text", "text_value"]:
            st.write(clean_result.get("text") or clean_result.get("value"))

        elif clean_result["type"] == "executive_summary":
            st.subheader("📌 Executive Summary")
            st.write(clean_result["summary_text"])
            if "rows" in clean_result:
                st.dataframe(clean_result["rows"], use_container_width=True)
            if "image_base64" in clean_result:
                st.image(base64.b64decode(clean_result["image_base64"]), caption="Summary Chart")

        else:
            st.write("ℹ️ No usable output returned.")
            st.json(clean_result)

    # No rerun needed — UI already refreshed



