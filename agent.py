import os, json, pandas as pd
from typing import TypedDict, Optional
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

load_dotenv()
model = init_chat_model("google_genai:gemini-3.1-flash-lite")

class PipelineState(TypedDict):
    raw_path: str
    df: Optional[pd.DataFrame]
    profile: str
    errors: list
    attempts: int
    status: str

def ingest(state):
    df = pd.read_csv(state["raw_path"])
    df.columns = [c.strip().replace("/", "_").replace(" ", "_").lower() for c in df.columns]
    profile = f"dtypes:\n{df.dtypes}\n\nnulls:\n{df.isna().sum()}\n\nsample:\n{df.head(5)}"
    return {"df": df, "profile": profile, "errors": [], "attempts": 0}

ALLOWED = {"strip_whitespace", "parse_dates", "fill_missing", "drop_duplicates", "cast_numeric", "drop_null_rows"}

def apply_ops(df, ops):
    log = []
    for op in ops:
        name, col = op.get("op"), op.get("column")
        if name not in ALLOWED:
            log.append(f"skipped unknown op: {op}")
        elif name == "drop_duplicates":
            df = df.drop_duplicates()
        elif col not in df.columns:
            log.append(f"skipped {name}: column '{col}' not found")
        elif name == "strip_whitespace":
            df[col] = df[col].astype("string").str.strip()
        elif name == "parse_dates":
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
        elif name == "cast_numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif name == "fill_missing":
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("Unknown")
        elif name == "drop_null_rows":
            df = df.dropna(subset=[col])
    return df, log

def clean(state):
    prompt = f"""You clean datasets. Return ONLY a JSON list of operations.
Allowed ops: strip_whitespace, parse_dates, fill_missing, drop_duplicates, cast_numeric, drop_null_rows.
Use the exact column names from the profile. Format:
[{{"op": "parse_dates", "column": "observationdate"}}, {{"op": "drop_duplicates"}}]
Use drop_null_rows on a column when its remaining null rows cannot be repaired (for example, unparseable dates).

Dataset profile:
{state['profile']}

Errors from the previous attempt (fix these): {state['errors']}"""
    resp = model.invoke(prompt)
    text = resp.content if isinstance(resp.content, str) else resp.content[0]["text"]
    ops = json.loads(text.replace("```json", "").replace("```", "").strip())
    print(f"[clean] attempt {state['attempts'] + 1} ops: {ops}")
    df, log = apply_ops(state["df"].copy(), ops)
    if log:
        print("[clean] warnings:", log)
    return {"df": df, "attempts": state["attempts"] + 1}

REQUIRED = ["observationdate", "country_region", "confirmed", "deaths", "recovered", "last_update"]

def validate(state):
    df, errors = state["df"], []
    if df.duplicated().any():
        errors.append("duplicate rows remain")
    for col in REQUIRED:
        if col not in df.columns:
            errors.append(f"required column '{col}' is missing")
        elif df[col].isna().any():
            errors.append(f"column '{col}' still has {int(df[col].isna().sum())} nulls")
    print(f"[validate] attempt {state['attempts']}: {errors if errors else 'passed'}")
    return {"errors": errors}

def load(state):
    url = URL.create("postgresql+psycopg", username=os.getenv("DB_USER"),
                     password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"),
                     port=int(os.getenv("DB_PORT")), database=os.getenv("DB_NAME"))
    state["df"].to_sql("covid_clean", create_engine(url), if_exists="replace", index=False)
    return {"status": "loaded"}

def route(state):
    if not state["errors"]:
        return "load"
    if state["attempts"] < 3:
        return "clean"
    return END

graph = StateGraph(PipelineState)
graph.add_node("ingest", ingest)
graph.add_node("clean", clean)
graph.add_node("validate", validate)
graph.add_node("load", load)

graph.add_edge(START, "ingest")
graph.add_edge("ingest", "clean")
graph.add_edge("clean", "validate")
graph.add_conditional_edges("validate", route)
graph.add_edge("load", END)

pipeline = graph.compile()

if __name__ == "__main__":
    raw = pd.read_csv("data/messy_covid.csv")
    print("BEFORE -> rows:", len(raw), "| null cells:", int(raw.isna().sum().sum()), "| duplicate rows:", int(raw.duplicated().sum()))

    result = pipeline.invoke({"raw_path": "data/messy_covid.csv"})
    print("attempts:", result["attempts"], "| status:", result.get("status", "failed"), "| errors:", result["errors"])

    out = result["df"]
    print("AFTER  -> rows:", len(out), "| null cells:", int(out.isna().sum().sum()), "| duplicate rows:", int(out.duplicated().sum()))