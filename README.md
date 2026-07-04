# VerifAI

VerifAI is an AI hallucination detector. Paste AI-generated text and it extracts individual claims, verifies them against external sources, and returns a trust score with annotated output.

Built with **LangGraph** (parallel claim verification), **Azure OpenAI**, and **FastAPI**.

## What it does

1. **Extract** — pulls verifiable claims from the input (stats, facts, legal citations, dates)
2. **Verify** — checks each claim in parallel using domain-specific sources
3. **Report** — computes a trust score and markdown summary
4. **Annotate** — returns the original text with inline verification tags

## Project structure

```
verifAI/
├── verifai.py          # LangGraph pipeline + FastAPI server
├── frontend/
│   └── index.html      # Web UI
├── Procfile            # Deployment process (Heroku / Railway / Render)
└── README.md
```

Environment variables are loaded from `../.env` (project root).

## Requirements

Install dependencies from the repo root:

```bash
pip install -r ../requirements.txt
```

Or from this folder:

```bash
pip install openai langgraph ddgs fastapi "uvicorn[standard]" pydantic
```

## Environment variables

Create `../.env` (or set these on your deployment platform):

```env
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
OPENAI_API_KEY=your-azure-api-key
OPENAI_API_VERSION=2025-04-01-preview
AZURE_DEPLOYMENT=gpt-4o
```

`AZURE_DEPLOYMENT` must match your Azure model deployment name exactly.

## Run locally

From the `verifAI/` directory:

```bash
python verifai.py
```

Or with uvicorn directly:

```bash
uvicorn verifai:app --host 0.0.0.0 --port 8000 --reload
```

Open the UI: [http://localhost:8000](http://localhost:8000)

## Web UI

- Paste AI-generated text
- Choose a domain: **Financial**, **Legal**, or **General**
- Click **Verify claims**
- Results show:
  - Overall trust score
  - Annotated text (verified / unverified / hallucinated tags)
  - Claim-by-claim breakdown with evidence and sources

Use **Load example** to try a sample financial text with known hallucinations.

## API

### `GET /health`

```bash
curl http://localhost:8000/health
```

### `POST /verify`

```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Apple reported record revenue of $119.6 billion in Q1 2024.",
    "domain": "financial"
  }'
```

**Request body**

| Field    | Type   | Description                                      |
|----------|--------|--------------------------------------------------|
| `text`   | string | AI-generated text to verify                      |
| `domain` | string | `"general"`, `"legal"`, or `"financial"` (default: `"general"`) |

**Response**

| Field                | Description                                      |
|----------------------|--------------------------------------------------|
| `trust_score`        | 0.0–1.0 (VERIFIED=1, UNVERIFIED=0.5, HALLUCINATED=0) |
| `claims_checked`     | Number of claims verified                        |
| `hallucinated_count` | Number of hallucinated claims                    |
| `report`             | Markdown trust report                            |
| `annotated_text`     | Original text with inline tags                   |
| `results`            | Array of per-claim results (status, evidence, source URL) |

## Verification by domain

| Domain      | Sources used                                              |
|-------------|-----------------------------------------------------------|
| `general`   | DuckDuckGo web search + LLM assessment                    |
| `legal`     | CourtListener API + web search fallback                   |
| `financial` | SEC EDGAR filings + web search for specific figures     |

Claims are verified **in parallel** via LangGraph's `Send` API — one branch per claim.

## LangGraph pipeline

```
extractor → dispatch (fan-out) → verify_single_claim (×N parallel)
                                        ↓
                              reporter → annotator → END
```

State uses `Annotated[list, operator.add]` on `results` so parallel branches merge cleanly. Nodes return only the fields they update (not full state) to avoid duplicate results.

## Deploy

The included `Procfile` starts the web server:

```
web: uvicorn verifai:app --host 0.0.0.0 --port ${PORT:-8000}
```

**Heroku / Railway / Render**

1. Set root directory to `verifAI` (or use `cd verifAI && ...` in the Procfile)
2. Set the Azure environment variables on the platform
3. Deploy — the platform runs the `web` process automatically

Run commands from `verifAI/` so relative paths (`frontend/`, `./memory`) resolve correctly.

## Notes

- Verification can take 30–90 seconds depending on claim count (parallel web/SEC lookups + LLM calls)
- Some websites block automated fetches (403) — the agent falls back to web search
- ChromaDB memory is not used in VerifAI; verification is stateless per request
