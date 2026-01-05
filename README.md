# Fake News Detection Platform

Multilingual, multimodal fact-checking system with retrieval‑augmented reasoning, on‑chain reputation, and a Gradio UI. Supports Bangla and English text, URLs, and images (OCR/caption → text).

## Features
- **Unified pipeline (text + image):** OCR/captioned images flow into the same verifier as text/URLs.
- **Retrieval‑augmented reasoning:** Multi-query search, evidence deduplication, factual summarization, LLM judgment (REAL/FAKE/MISINFORMATION/UNSURE) with credibility score and short rationale.
- **Reputation ledger:** URL and publisher lookups, publisher flag count, and on‑chain registration for flagged cases; transaction hash surfaced in UI.
- **Local traceability:** Claim records stored as JSON with evidence, flagged sources, and on‑chain metadata.
- **Gradio UI:** Shows verdict, score, evidence list, publisher reputation count, and blockchain hash.

## High-Level Flow
1) **Input:** User submits text/URL or image.  
2) **Prep:** Clean/normalize text; fetch URL content; OCR/caption images.  
3) **Retrieve:** Generate multiple queries; collect and deduplicate evidence.  
4) **Summarize:** Reduce evidence to short factual notes.  
5) **Judge:** LLM produces label, score (0–100), and explanation.  
6) **Reputation:** For FAKE/MISINFO, flag sources; check URL/publisher history; register on-chain; capture tx hash and publisher count.  
7) **Store & Display:** Persist JSON record; render UI cards with verdict, evidence, reputation, and hash.

## Key Modules
- `code/fact_check_llm.py`: Core verification pipeline (text + image via shared flow), retrieval, summarization, judgment, storage, on‑chain metadata.
- `code/image_fact_checker.py`: Image ingestion, OCR, captioning, and handoff to verifier.
- `code/blockchain_registry.py`: Lightweight client for ledger interactions (register, lookup URL, lookup publisher).
- `code/claim_storage.py`: JSON storage for claims, evidence, flagged sources, and on‑chain data.
- `code/app.py`: Gradio UI wiring, presentation of verdict/evidence/reputation/tx hash.
- `code/architecture_diagram.py` / `code/architecture.mmd`: Mermaid architecture diagram generator/output.

## Tests
- `code/test_blockchain_registry.py`: Mocks ledger client calls.
- `code/test_claim_storage.py`: Validates storage, embeddings, snapshots, and on‑chain metadata field.
- `code/test_image_verification.py` / others: Cover image/text verification helpers.

## Running the UI
```bash
python code/app.py
```

## Environment Notes
- Configure required model and key settings via `.env` alongside code (LLM, translation, search, ledger settings).
- For blockchain: ensure RPC, contract address, and keys are set on the backend service that signs transactions; this client uses the hosted bridge.

## Architecture Snapshot
Mermaid: see `code/architecture.mmd`. Generate fresh via:
```bash
python code/architecture_diagram.py
```
