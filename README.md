# The Archive — Student Final-Year Project Repository

A one-week prototype for MCS 306 that goes beyond "upload and browse" by adding
three things a real institutional archive needs: **search that understands
meaning**, **automatic overlap/plagiarism flagging**, and a **departmental
decision-support dashboard**.

## Why this is more than a file store

| Feature | What it does | How |
|---|---|---|
| Semantic search | Searching "queue systems" surfaces "Hospital Wait Time Optimizer" even with zero keyword overlap | TF-IDF vectorization of title+abstract+keywords, ranked by cosine similarity |
| Overlap detection | On submission, flags projects that closely resemble existing work before it's saved | Same TF-IDF/cosine pipeline, applied at submit-time against the whole corpus |
| Related projects | Every project page recommends conceptually similar past projects | Cosine similarity against the same corpus, excluding self |
| Lineage tracking | Students can declare "builds on project X," creating a visible research thread across cohorts | Self-referencing FK (`builds_on_id`) rendered as a chain on the detail page |
| Supervisor & topic dashboard | Shows supervisor load and topic-area saturation so gaps in coverage are visible | Aggregate SQL queries rendered as bar charts |

## Running it

```bash
pip install -r requirements.txt
python seed.py     # populates realistic sample data (15 projects, real overlap clusters)
python app.py       # http://localhost:5000
```

Skip `seed.py` if you want to start from an empty archive — `app.py` will
create the schema automatically on first run.

## Architecture

- **Backend:** Flask, single `app.py`, SQLite (`repository.db`)
- **Search/similarity engine:** scikit-learn `TfidfVectorizer` + `cosine_similarity`,
  rebuilt on each request from the live corpus — no external API or vector DB
  needed for this scale, and it's fully explainable in a viva ("here's exactly
  why it matched")
- **Auto-summary:** lightweight extractive summarizer (first informative
  sentences) — the code marks exactly where a Groq/Claude API call would slot
  in for true abstractive summaries in a production version
- **Frontend:** server-rendered Jinja templates, no JS framework, single
  `style.css` — "reading room" archive-ledger visual identity (accession
  numbers, lineage chains, similarity stamps) rather than a generic admin
  panel look

## What to demo live

1. Search `"queue systems"` on the homepage — show it surfacing the wait-time
   project without keyword overlap.
2. Go to **Deposit a Project**, submit an abstract closely paraphrasing the
   Rural Clinic Queue Tracker — show the overlap warning firing before save.
3. Open **Hospital Wait Time Optimizer** — show the lineage chain back to the
   Rural Clinic Queue Tracker, and the related-projects panel.
4. Open **Ledger** — show supervisor load and topic coverage, and the
   overlap-flags table populated from the seed data.

## Extending beyond the prototype

- Swap `auto_summary()` for a real Groq/Claude API call for abstractive
  (not extractive) summaries
- Swap TF-IDF for sentence embeddings (e.g. `sentence-transformers`) for
  stronger semantic matching at larger corpus sizes
- Add auth so supervisors can only edit their own students' entries
- Move file storage to Supabase/S3 instead of local disk for multi-instance
  deployment
