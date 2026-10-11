# Agentic Data Pipeline Assistant

A self-correcting data pipeline built with LangGraph. An LLM (Google Gemini) decides how to clean a messy CSV, deterministic code applies the cleaning steps, a validator checks the result, and any failure loops back to the LLM with the error message so it can try again (up to 3 attempts) before the data is loaded into PostgreSQL.

## What it does

1. **Ingest:** reads a messy CSV, standardizes column names (lowercase, snake_case), and builds a profile of the data (column types, null counts, sample rows)
2. **Clean (LLM):** the LLM reads the profile and any errors from the previous attempt, then returns a JSON list of cleaning operations
3. **Validate:** checks for duplicate rows and for missing values in required columns
4. **Route:** if validation passes, the data is loaded; if it fails and attempts remain, the pipeline loops back to the clean step with the error details; after 3 failed attempts it stops without loading
5. **Load:** writes the cleaned data to a PostgreSQL table (`covid_clean`)

```
ingest → clean (LLM) → validate ──pass──► load → END
            ▲              │
            └──── fail ────┘   (max 3 attempts, then stop)
```

## Why I built it this way

The LLM never writes or runs code. It can only choose from a fixed set of operations, and my own code applies them:

`strip_whitespace`, `parse_dates`, `fill_missing`, `drop_duplicates`, `cast_numeric`, `drop_null_rows`

This keeps the agent's behavior predictable and safe, while the LLM still makes the decisions about *which* operations to use on *which* columns. Validation is plain Python (not an LLM), so a pass or fail is deterministic. The retry loop gives the agent the validator's error message so it can correct its own plan.

## Tech stack

- **Orchestration:** LangGraph (`StateGraph` with a conditional edge for the retry loop)
- **LLM:** Google Gemini (`gemini-3.1-flash-lite`) via LangChain's `init_chat_model`
- **Data handling:** pandas
- **Database:** PostgreSQL, loaded with SQLAlchemy and `psycopg`
- **Dataset:** COVID-19 daily case data from Kaggle (`covid_19_data.csv`), with deliberately injected data-quality problems
- **Env management:** `uv`

## Setup

**1. Install dependencies**
```bash
uv sync
```

**2. Configure environment variables**

Copy `.env.example` to `.env` and fill in your values:
```
GOOGLE_API_KEY=your_gemini_key
DB_NAME=covid_db
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
```

**3. Generate the messy dataset**

Place `covid_19_data.csv` in the project root, then run:
```bash
uv run main.py
```
This samples 5,000 rows and injects problems: missing `Confirmed` values, invalid dates, whitespace-padded country names, and 100 duplicated rows. The result is saved to `data/messy_covid.csv`.

**4. Run the pipeline**
```bash
uv run agent.py
```

## Sample run

```
BEFORE -> rows: 5100 | null cells: 1580 | duplicate rows: 100
[validate] attempt 1: ["column 'observationdate' still has 190 nulls"]
[validate] attempt 2: passed
attempts: 2 | status: loaded | errors: []
AFTER  -> rows: 4510 | null cells: 0 | duplicate rows: 0
```

What happened: on attempt 1 the LLM parsed the dates, but 190 values could not be parsed and became nulls. Validation rejected the result and passed the error back. On attempt 2 the LLM added a `drop_null_rows` operation for `observationdate`, validation passed, and the data was loaded.

| Metric | Before | After |
|---|---|---|
| Rows | 5,100 | 4,510 |
| Duplicate rows | 100 | 0 |
| Null cells (all columns) | 1,580 | 0 |
| Nulls in required columns | not checked | 0 (enforced by validation) |

You can verify the load directly in Postgres:
```sql
SELECT COUNT(*) FROM covid_clean;
SELECT COUNT(*) - COUNT(DISTINCT sno) AS duplicate_rows FROM covid_clean;
SELECT COUNT(*) FROM covid_clean WHERE confirmed IS NULL OR observationdate IS NULL;
```

## Known limitations

- **Rows are dropped, not repaired, for bad dates and missing counts.** Rows with an unparseable date or a missing `confirmed` value are removed, so some data is lost (590 of 5,100 rows in the sample run, including the 100 duplicates).
- **Validation covers six of eight columns.** Required columns are `observationdate`, `country_region`, `confirmed`, `deaths`, `recovered` and `last_update`. `sno` and `province_state` are not checked for nulls (`province_state` is filled with "Unknown" by the cleaning step).
- **Fill values are simple.** Missing numeric values are filled with 0 and missing text with "Unknown". For cumulative counts, 0 can understate totals; forward-filling within each country would be better.
- **The LLM's reply must be valid JSON.** There is no fallback if the model returns malformed output.
- **Fixed retry limit.** After 3 failed attempts the pipeline stops without loading anything.
- **No automated tests yet.**

## Possible improvements

- Add a `fill_forward` operation (grouped by country) for cumulative columns
- Handle malformed LLM output with a retry or a fallback
- Log each run (attempts, rows removed) to a Postgres table
- Wrap the pipeline in a FastAPI endpoint
- Add unit tests for each cleaning operation and the validator