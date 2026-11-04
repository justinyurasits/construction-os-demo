# Construction OS — Demo

A working demo of Construction OS: a Flask application backed by a ChromaDB vector store, with the construction document corpus embedded and queryable via the Claude API.

This is the demo environment used for client walkthroughs — not the production system.

## What's here

- **`app.py`** — Flask application. Handles document queries against the embedded corpus, change order drafting, and session management.
- **`ingest_construction_os.py`** — Ingestion pipeline: reads the document corpus, chunks it, and loads it into ChromaDB.
- **`job_cost_report.py`** — Job cost reporting tool: reads receipts and produces a formatted budget-vs-actual report.
- **`budget_summary.py`** — Budget summary utility.
- **`construction_os_db/`** — ChromaDB vector store with the corpus embedded. Regenerate with `ingest_construction_os.py`.
- **`construction-os-metrics.html`** — Static metrics page used in demos.

## Stack

- Python / Flask
- ChromaDB (local vector store)
- Claude API (Anthropic) — document Q&A, change order generation
- `python-docx` / `openpyxl` for document output

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# (Optional) Re-ingest the corpus
python ingest_construction_os.py

# Run the demo
python app.py
```

Open `http://localhost:5000`.

## Notes

The vector store in `construction_os_db/` is pre-built from the Construction OS document corpus. If you modify the corpus, re-run `ingest_construction_os.py` to rebuild it.
