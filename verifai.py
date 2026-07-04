import json
import os
import time
import re
from datetime import date
from pathlib import Path
from typing import TypedDict, Annotated
import operator

from openai import AzureOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from ddgs import DDGS
import urllib.request
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── setup ────────────────────────────────────────────────────────
env_file = Path(__file__).resolve().parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

MODEL = os.environ.get("AZURE_DEPLOYMENT", "gpt-4o")
client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_key=os.environ["OPENAI_API_KEY"],
    api_version=os.environ["OPENAI_API_VERSION"],
)


def llm(messages: list, model: str = MODEL) -> str:
    response = client.chat.completions.create(
        model=model, max_tokens=2048, messages=messages, temperature=0
    )
    return response.choices[0].message.content


# ════════════════════════════════════════════════════════════════
# STATE — the object that flows through every node
# ════════════════════════════════════════════════════════════════


class VerificationState(TypedDict):
    input_text: str
    domain: str
    claims: list[dict]
    results: Annotated[list[dict], operator.add]  # ← add reducer
    report: str
    trust_score: float
    annotated_text: str

# ════════════════════════════════════════════════════════════════
# NODE 1 — CLAIM EXTRACTOR
# ════════════════════════════════════════════════════════════════

EXTRACTOR_PROMPT = """You are a claim extraction specialist.

Given a piece of AI-generated text, extract every verifiable claim and citation.

CRITICAL RULES:
- Extract ALL legal case citations — even if they appear fictional or uncertain
- The purpose is to VERIFY whether they are real — that is not your job to decide
- Extract ALL statistics and numbers, even if they seem plausible
- Do NOT skip any claim — missing a hallucinated claim is a critical failure

For each claim output a JSON object with:
- id: sequential number starting at 1
- text: the exact claim as stated
- type: one of "legal_citation" | "statistic" | "factual" | "date"
- source_sentence: the full sentence it came from

Return ONLY a JSON array. No explanation. No markdown."""


def extract_claims(state: VerificationState) -> VerificationState:
    print("\n[Extractor] Pulling claims from text...")

    response = llm(
        [
            {"role": "system", "content": EXTRACTOR_PROMPT},
            {"role": "user", "content": state["input_text"]},
        ]
    )

    print(f"\n[Extractor] Raw LLM response:\n{response}\n")  # add this line
    try:
        clean = response.strip()
        if clean.startswith("```json") and clean.endswith("```"):
            clean = clean[7:-3]
        if clean.startswith("```") and clean.endswith("```"):
            clean = clean[3:-3]
        if clean.startswith("json") and clean.endswith("json"):
            clean = clean[4:-4]
        if clean.startswith("json") and clean.endswith("json"):
            clean = clean[4:-4]
        claims = json.loads(clean)
    except Exception as e:
        print(f"  [Extractor] Parse error: {e}")
        claims = []

    print(f"  [Extractor] Found {len(claims)} claims")
    for c in claims:
        print(f"    [{c['type']}] {c['text'][:80]}")

    return {"claims": claims}


# ════════════════════════════════════════════════════════════════
# NODE 2 — FACTUAL VERIFIER (web search based)
# ════════════════════════════════════════════════════════════════


def search_web(query: str, max_results: int = 3) -> str:
    results = []
    for _ in range(3):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if results:
                break
        except Exception:
            pass
        time.sleep(1)
    if not results:
        return "No results found."
    return "\n\n".join(
        f"{i}. {r['title']}\n   {r['href']}\n   {r['body']}"
        for i, r in enumerate(results, 1)
    )


VERIFIER_PROMPT = """You are a fact-checking agent.

You will receive a claim and web search results.
Determine whether the claim is supported, contradicted, or unverifiable.

Output a JSON object with:
- status: "VERIFIED" | "HALLUCINATED" | "UNVERIFIED"
- confidence: 0.0 to 1.0
- evidence: one sentence explaining your verdict with source if available
- source_url: the URL that best supports your verdict, or null

Rules:
- VERIFIED: search results clearly support the claim
- HALLUCINATED: search results clearly contradict the claim
- UNVERIFIED: search results are inconclusive or not found

Return ONLY the JSON object. No explanation. No markdown."""


def verify_single_claim(state: VerificationState) -> dict:
    """
    Verifies exactly one claim. Designed to run in parallel with other instances.
    Each parallel branch receives a state where claims contains exactly one claim.
    """
    claim = state["claims"][0]  # only one claim in this branch's state
    domain = state["domain"]

    print(f"  [verify] [{claim['type']}] {claim['text'][:60]}")

    if claim["type"] == "legal_citation":
        result = verify_legal_citation(claim)
    elif claim["type"] == "statistic" and domain == "financial":
        result = verify_financial_claim(claim)
    else:
        result = verify_factual_claim(claim)

    print(f"  [verify] → {result['status']} ({result['confidence']:.0%})")
    return {"results": [result]}


def verify_factual_claim(claim: dict) -> dict:
    """Verify a single factual or statistical claim via web search."""
    search_results = search_web(claim["text"])

    response = llm(
        [
            {"role": "system", "content": VERIFIER_PROMPT},
            {
                "role": "user",
                "content": f"""
Claim: {claim['text']}

Search results:
{search_results}

Verdict:""",
            },
        ]
    )

    # TODO 2: parse the JSON response
    # Return a dict with claim_id, status, evidence, confidence, source_url
    try:
        clean = response.strip()
        if clean.startswith("```json") and clean.endswith("```"):
            clean = clean[7:-3]
        if clean.startswith("```") and clean.endswith("```"):
            clean = clean[3:-3]
        if clean.startswith("json") and clean.endswith("json"):
            clean = clean[4:-4]
        if clean.startswith("json") and clean.endswith("json"):
            clean = clean[4:-4]
        verdict = json.loads(clean)
        verdict["claim_id"] = claim["id"]
        verdict["claim_text"] = claim["text"]
        return verdict
    except Exception:
        return {
            "claim_id": claim["id"],
            "claim_text": claim["text"],
            "status": "UNVERIFIED",
            "confidence": 0.0,
            "evidence": "Could not parse verifier response",
            "source_url": None,
        }


# ════════════════════════════════════════════════════════════════
# NODE 3 — LEGAL CITATION VERIFIER
# ════════════════════════════════════════════════════════════════


def verify_legal_citation(claim: dict) -> dict:
    print(f"    [Legal] Checking: {claim['text'][:60]}")

    query = claim["text"].replace(" ", "+")
    url = f"https://www.courtlistener.com/api/rest/v4/search/?q={query}&type=o&format=json"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VerifAI/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())

        count = data.get("count", 0)

        if count > 0:
            top = data["results"][0]
            return {
                "claim_id": claim["id"],
                "claim_text": claim["text"],
                "status": "VERIFIED",
                "confidence": 0.9,
                "evidence": f"Found in CourtListener: {top.get('caseName', 'Unknown')} — {top.get('dateFiled', '')}",
                "source_url": f"https://www.courtlistener.com{top.get('absolute_url', '')}",
            }

        # Not in CourtListener — ask LLM to assess web search results
        search_result = search_web(f"court case ruling {claim['text']}")

        assessment = llm(
            [
                {
                    "role": "system",
                    "content": """You are a legal fact-checker.
You will receive a legal citation claim and web search results.
Determine if this is a real court case or a hallucinated/fictional one.

A real case will have:
- Multiple credible legal sources citing it
- Official court documents or legal databases mentioning it
- Consistent facts across sources

A hallucinated case will have:
- No direct mentions of the actual case
- Only tangentially related results
- No official court records

Output ONLY a JSON object:
{"status": "VERIFIED"|"HALLUCINATED"|"UNVERIFIED", "confidence": 0.0-1.0, "evidence": "one sentence explanation"}""",
                },
                {
                    "role": "user",
                    "content": f"""
Legal citation claim: {claim['text']}

Web search results:
{search_result}

Is this a real case or hallucinated?""",
                },
            ]
        )

        try:
            clean = re.sub(r"```json|```", "", assessment).strip()
            verdict = json.loads(clean)
            verdict["claim_id"] = claim["id"]
            verdict["claim_text"] = claim["text"]
            verdict["source_url"] = None
            return verdict
        except Exception:
            return {
                "claim_id": claim["id"],
                "claim_text": claim["text"],
                "status": "UNVERIFIED",
                "confidence": 0.3,
                "evidence": "Not found in CourtListener, web assessment failed",
                "source_url": None,
            }

    except Exception as e:
        return {
            "claim_id": claim["id"],
            "claim_text": claim["text"],
            "status": "UNVERIFIED",
            "confidence": 0.0,
            "evidence": f"Verification failed: {e}",
            "source_url": None,
        }


# ════════════════════════════════════════════════════════════════
# NODE 4 — VERIFICATION DISPATCHER
# Runs the right verifier for each claim type
# ════════════════════════════════════════════════════════════════


def dispatch_verification(state: VerificationState) -> list:
    print(f"\n[dispatch] Sending {len(state['claims'])} claims to parallel verification...")
    return [
        Send("verify_single_claim", {
            "input_text": state["input_text"],
            "domain": state["domain"],
            "claims": [claim],
            "results": [],
            "report": "",
            "trust_score": 0.0,
            "annotated_text": "",
        })
        for claim in state["claims"]
    ]

# ════════════════════════════════════════════════════════════════
# NODE 5 — TRUST SCORER + REPORT GENERATOR
# ════════════════════════════════════════════════════════════════
def generate_report(state: VerificationState) -> VerificationState:
    print("\n[Reporter] Generating trust report...")

    # Calibrate confidence scores first
    results = [calibrate_confidence(r) for r in state["results"]]
    if not results:
        return {"report": "No verifiable claims found.", "trust_score": 1.0}

    # Calculate trust score
    # TODO 4: calculate score
    # VERIFIED = 1.0 points, UNVERIFIED = 0.5 points, HALLUCINATED = 0.0 points
    # trust_score = total points / total claims
    score_map = {"VERIFIED": 1.0, "UNVERIFIED": 0.5, "HALLUCINATED": 0.0}
    trust_score = sum(score_map[r["status"]] for r in results) / len(results)

    # Build the report
    verified = [r for r in results if r["status"] == "VERIFIED"]
    unverified = [r for r in results if r["status"] == "UNVERIFIED"]
    hallucinated = [r for r in results if r["status"] == "HALLUCINATED"]

    # Emoji indicators
    status_icon = {"VERIFIED": "🟢", "UNVERIFIED": "🟡", "HALLUCINATED": "🔴"}

    lines = [
        f"# VerifAI Trust Report",
        f"**Date**: {date.today()}",
        f"**Overall Trust Score**: {trust_score:.0%}",
        f"**Claims checked**: {len(results)} total — "
        f"🟢 {len(verified)} verified · "
        f"🟡 {len(unverified)} unverified · "
        f"🔴 {len(hallucinated)} hallucinated",
        "",
        "---",
        "",
        "## Claim-by-Claim Results",
    ]

    for r in results:
        icon = status_icon[r["status"]]
        lines.append(f"\n{icon} **Claim {r['claim_id']}**: {r['claim_text']}")
        lines.append(
            f"   - Status: **{r['status']}** ({r['confidence']:.0%} confidence)"
        )
        lines.append(f"   - Evidence: {r['evidence']}")
        if r.get("source_url"):
            lines.append(f"   - Source: {r['source_url']}")

    if hallucinated:
        lines += [
            "",
            "---",
            "",
            "## ⚠️ Hallucinated Claims — Do Not Use",
        ]
        for r in hallucinated:
            lines.append(f"- Claim {r['claim_id']}: {r['claim_text']}")
            lines.append(f"  Evidence: {r['evidence']}")

    report = "\n".join(lines)
    print(f"  Trust score: {trust_score:.0%}")
    return {"report": report, "trust_score": trust_score}


import urllib.parse


_edgar_cache = {}

def query_edgar(company: str) -> tuple[list, int]:
    """Cached EDGAR lookup — avoids duplicate requests for the same company."""
    if company in _edgar_cache:
        return _edgar_cache[company]

    time.sleep(0.5)
    query_params = {
        "q": f'"{company}"',
        "forms": "10-K,10-Q",
        "dateRange": "custom",
        "startdt": "2023-01-01",
        "enddt": "2026-12-31"
    }
    search_url = "https://efts.sec.gov/LATEST/search-index?" + urllib.parse.urlencode(query_params)

    try:
        req = urllib.request.Request(search_url, headers={"User-Agent": "VerifAI Research research@verifai.com"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        hits = data.get("hits", {}).get("hits", [])
        total = data.get("hits", {}).get("total", {}).get("value", 0)
        relevant_hits = [
            h for h in hits
            if company.lower() in h.get("_source", {}).get("display_names", [""])[0].lower()
        ]
        result = (relevant_hits, total)
    except Exception as e:
        print(f"    [Financial] EDGAR request failed: {e}")
        result = ([], 0)

    _edgar_cache[company] = result
    return result


def verify_financial_claim(claim: dict, fallback_company: str = None) -> dict:
    print(f"    [Financial] >>> ENTERED verify_financial_claim for: {claim['text'][:50]}")
    context_text = claim.get("source_sentence", claim["text"])

    extraction = llm([
        {"role": "system", "content": """Extract the company name and financial metric from this claim.
Output ONLY JSON: {"company": "company name", "metric": "what is being claimed", "value": "the number claimed"}
If you cannot determine a company name even from context, output: {"company": null, "metric": null, "value": null}"""},
        {"role": "user", "content": context_text}
    ])

    try:
        clean = re.sub(r"```json|```", "", extraction).strip()
        parsed = json.loads(clean)
    except Exception:
        parsed = {"company": None, "metric": None, "value": None}

    company = parsed.get("company") or fallback_company

    if not company:
        result = verify_factual_claim(claim)
        result["company_used"] = None
        return result

    try:
        relevant_hits, total = query_edgar(company)
        print(f"    [Financial] EDGAR for '{company}': {total} total, {len(relevant_hits)} relevant")

        if relevant_hits:
            filing_snippets = "\n".join([
                f"- {h['_source'].get('display_names', ['Unknown'])[0]}: {h['_source'].get('file_date', '')} ({h['_source'].get('form_type', h['_source'].get('forms', ''))})"
                for h in relevant_hits[:5]
            ])

            search_query = f"{company} {claim['text']}"
            web_results = search_web(search_query)

            assessment = llm([
                {"role": "system", "content": """You are a financial fact-checker.

You have two sources of evidence:
1. SEC EDGAR filing records — these PROVE the company is real and actively files
   with regulators. They do NOT contain the specific number being claimed.
2. Web search results — these contain actual reported figures and should be your
   PRIMARY source for verifying the specific number in the claim.

Reasoning process:
- If SEC filings confirm the company exists AND web search confirms the specific number → VERIFIED
- If SEC filings confirm the company exists BUT web search contradicts the specific number → HALLUCINATED
- If SEC filings confirm the company exists AND web search is silent on this specific number → UNVERIFIED
- If SEC filings show no matching company → HALLUCINATED

revenue, net income, earnings, and profit margin are NOT interchangeable —
if the claim's metric doesn't match what the sources report, this is inaccurate.

Output ONLY JSON:
{"status": "VERIFIED"|"HALLUCINATED"|"UNVERIFIED", "confidence": 0.0-1.0, "evidence": "one sentence citing the specific number found", "source_url": "the most relevant URL from web search, or null"}"""},
                {"role": "user", "content": f"""
Financial claim: {claim['text']}
Company: {company}
Claimed metric: {parsed.get('metric')}
Claimed value: {parsed.get('value')}

SEC EDGAR confirms company exists and files regularly: {filing_snippets}

Web search results: {web_results}

What is your verdict?"""}
            ])

            clean = re.sub(r"```json|```", "", assessment).strip()
            verdict = json.loads(clean)
            verdict["claim_id"] = claim["id"]
            verdict["claim_text"] = claim["text"]
            verdict["company_used"] = company
            return verdict

        else:
            search_query = f"{company} {claim['text']}"
            web_results = search_web(search_query)
            print(f"    [Financial] No matching SEC filings, web search done for '{company}'")

            assessment = llm([
                {"role": "system", "content": """You are a financial fact-checker.

A company claim was checked against SEC EDGAR and NO MATCHING filings were found
under that exact company name. This is a strong signal the company either does not
exist, is not publicly traded, or the claim is fabricated.

Use web search results to make a final judgment.
If web search also finds little to nothing specific about this company's financials,
mark HALLUCINATED. If web search shows the company is real but privately held or
non-US, mark UNVERIFIED.

Output ONLY JSON:
{"status": "VERIFIED"|"HALLUCINATED"|"UNVERIFIED", "confidence": 0.0-1.0, "evidence": "one sentence", "source_url": null}"""},
                {"role": "user", "content": f"""
Financial claim: {claim['text']}
Company: {company}
SEC EDGAR: no filings found under this exact name

Web search results:
{web_results}

Does this company exist? Is this claim accurate?"""}
            ])

            try:
                clean = re.sub(r"```json|```", "", assessment).strip()
                verdict = json.loads(clean)
                verdict["claim_id"] = claim["id"]
                verdict["claim_text"] = claim["text"]
                verdict["company_used"] = company
                return verdict
            except Exception:
                result = verify_factual_claim(claim)
                result["company_used"] = company
                return result

    except Exception as e:
        print(f"    [Financial] Verification failed: {e}")
        result = verify_factual_claim(claim)
        result["company_used"] = company
        return result


def verify_all_claims(state: VerificationState) -> VerificationState:
    print(f"\n[Verifier] Checking {len(state['claims'])} claims...")

    results = []
    last_company = None  # track company across claims

    for claim in state["claims"]:
        print(f"  Verifying [{claim['type']}]: {claim['text'][:60]}")

        if claim["type"] == "legal_citation":
            result = verify_legal_citation(claim)
        elif claim["type"] == "statistic" and state["domain"] == "financial":
            result = verify_financial_claim(claim, fallback_company=last_company)
            if result.get("company_used"):
                last_company = result["company_used"]
        else:
            result = verify_factual_claim(claim)

        print(f"    → {result['status']} ({result['confidence']:.0%})")
        results.append(result)

    return {**state, "results": results}


SOURCE_QUALITY = {
    # High authority sources
    "courtlistener.com": 0.95,
    "sec.gov": 0.98,
    "apple.com": 0.98,
    "supremecourt.gov": 0.99,
    "pubmed.ncbi.nlm.nih.gov": 0.95,
    # News and reference
    "reuters.com": 0.85,
    "bloomberg.com": 0.85,
    "ft.com": 0.85,
    "wikipedia.org": 0.70,
    # General web
    "default": 0.60,
}


def calibrate_confidence(result: dict) -> dict:
    """Adjust confidence based on source quality."""
    if result["status"] == "HALLUCINATED":
        # Hallucinated confidence stays as-is — it represents how sure we are it's fake
        return result

    source_url = result.get("source_url") or ""
    quality = SOURCE_QUALITY["default"]

    for domain, score in SOURCE_QUALITY.items():
        if domain in source_url:
            quality = score
            break

    # Blend LLM confidence with source quality
    original = result.get("confidence", 0.5)
    calibrated = (original * 0.6) + (quality * 0.4)
    result["confidence"] = round(calibrated, 2)
    return result

def generate_annotated_text(state: VerificationState) -> VerificationState:
    """
    Produces the original text with inline annotations.
    Hallucinated claims get [🔴 HALLUCINATED] tags.
    Verified claims get [🟢] tags.
    """
    print(f"\n[Annotator] >>> ENTERED generate_annotated_text")  # add this
    print(f"[Annotator] state has {len(state.get('results', []))} results")  # add this

    if not state["results"]:
        print(f"[Annotator] No results — returning early without annotating")
        return {}

    annotated = state["input_text"]

    lookup = {r["claim_text"]: r for r in state["results"]}

    annotation_prompt = f"""You are a text annotator.

Given the original text and a list of verified/hallucinated claims,
rewrite the text with inline annotations after each claim.

Use these exact tags:
- After a VERIFIED claim: [🟢 VERIFIED]
- After a HALLUCINATED claim: [🔴 HALLUCINATED — {"{evidence}"}]
- After an UNVERIFIED claim: [🟡 UNVERIFIED]

Return the full annotated text. Do not change any wording — only add the tags.

Original text:
{state["input_text"]}

Claims and their verdicts:
{json.dumps([{"text": r["claim_text"], "status": r["status"], "evidence": r["evidence"]} for r in state["results"]], indent=2)}

Return the annotated text:"""

    annotated = llm(
        [
            {
                "role": "system",
                "content": "You annotate text with verification results. Return only the annotated text.",
            },
            {"role": "user", "content": annotation_prompt},
        ]
    )

    print(f"[Annotator] Generated text, length: {len(annotated)}")  # add this
    print(f"[Annotator] Preview: {annotated[:200]}")  # add this

    return {"annotated_text": annotated}

# ════════════════════════════════════════════════════════════════
# LANGGRAPH — wire the nodes together
# ════════════════════════════════════════════════════════════════


def build_graph():
    graph = StateGraph(VerificationState)

    graph.add_node("extractor",          extract_claims)
    graph.add_node("verify_single_claim", verify_single_claim)
    graph.add_node("reporter",           generate_report)
    graph.add_node("annotator",          generate_annotated_text)

    graph.set_entry_point("extractor")

    # Fan-out: extractor → dispatch → many verify_single_claim nodes running in parallel
    graph.add_conditional_edges(
        "extractor",
        dispatch_verification,
        ["verify_single_claim"]  # all branches go to this node
    )

    # Fan-in: all verify_single_claim results collected → reporter
    graph.add_edge("verify_single_claim", "reporter")
    graph.add_edge("reporter",            "annotator")
    graph.add_edge("annotator",           END)

    return graph.compile()


verifier_graph = build_graph()


# ════════════════════════════════════════════════════════════════
# FASTAPI
# ════════════════════════════════════════════════════════════════

app = FastAPI(title="VerifAI — AI Hallucination Detector")


class VerifyRequest(BaseModel):
    text: str
    domain: str = "general"  # "legal", "financial", "general"


class ClaimResult(BaseModel):
    claim_id: int | str
    claim_text: str
    status: str
    confidence: float
    evidence: str
    source_url: str | None = None


class VerifyResponse(BaseModel):
    trust_score: float
    claims_checked: int
    hallucinated_count: int
    report: str
    annotated_text: str
    results: list[ClaimResult]

@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    global _edgar_cache
    _edgar_cache = {}   # ← clear cache between requests

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    initial_state: VerificationState = {
    "input_text": req.text,
    "domain": req.domain,
    "claims": [],
    "results": [],      # ← this must be [] not carrying over from previous request
    "report": "",
    "trust_score": 0.0,
    "annotated_text": "",
    }

    final_state = verifier_graph.invoke(initial_state)

    hallucinated = [r for r in final_state["results"] if r["status"] == "HALLUCINATED"]

    return VerifyResponse(
        trust_score=final_state["trust_score"],
        claims_checked=len(final_state["results"]),
        hallucinated_count=len(hallucinated),
        report=final_state["report"],
        annotated_text=final_state.get("annotated_text", ""),
        results=final_state["results"],
    )


@app.get("/health")
def health():
    return {"status": "ok", "date": str(date.today())}

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add this after creating the app
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

# ════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
