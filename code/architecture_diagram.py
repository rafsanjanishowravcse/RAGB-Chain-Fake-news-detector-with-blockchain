"""
Generate a Mermaid diagram for the current system architecture.

Focus:
- Bangla + English text pipeline 
- Groq LLM blocks shown as placeholders for your fine-tuned model.
- Image flow uses OCR/captioning to feed text into the same verifier.
- Blockchain registry used for URL + publisher reputation and on-chain tx hash.
"""
from pathlib import Path


def build_mermaid() -> str:
    # Mermaid kept inline so it can be rendered with mmdc, mermaid-cli, or in Markdown viewers.
    return """```mermaid
flowchart LR
    %% Client Layer
    user[User (Bangla/English text / image)]
    ui[Gradio UI (app.py)]

    %% Text Verification Core
    verify[fact_check_llm.verify_news]
    qgen[Groq LLM\\nMulti-query gen\\n(placeholder: your fine-tuned model)]
    search[Serper.dev search\\n+ semantic reranker]
    summarize[Groq LLM\\nEvidence summarizer\\n(placeholder model)]
    judge[Groq LLM\\nVerdict + credibility\\n(placeholder model)]
    storage[ClaimStorage JSON + embeddings\\n(code/claim_metadata)]
    onchain_meta[On-chain metadata\\n(reputation + tx hash)]
    ui_out[Result cards\\n(REAL/FAKE/UNSURE + score)]

    %% Image Flow
    img_checker[ImageFactChecker\\n(image_fact_checker.py)]
    ocr[OCR + captioning\\n(EasyOCR/Tesseract, BLIP/CLIP if available)]

    %% Language Support
    translate[SarvamAI\\ntranslate (non-Bengali)]

    %% Blockchain Registry
    registry[Blockchain Registry API\\n(fakensethfa.onrender.com)]
    register[POST /register\\n(flagged URL)]
    lookup_url[GET /getNews?url=...]
    lookup_pub[GET /getNewsByPublisher?publisher=...]

    %% Relations
    user --> ui
    ui -- "Text input" --> verify
    verify --> translate
    verify --> qgen --> search --> summarize --> judge
    judge --> lookup_url --> registry
    judge --> lookup_pub --> registry
    judge --> register --> registry
    judge --> onchain_meta
    judge --> storage
    judge --> ui_out
    ui_out --> ui

    %% Image path reuses text verifier
    ui -- "Image upload" --> img_checker --> ocr --> verify
```
"""


def main() -> None:
    mermaid_diagram = build_mermaid()
    out_path = Path(__file__).parent / "architecture.mmd"
    out_path.write_text(mermaid_diagram, encoding="utf-8")
    print(f"Wrote Mermaid diagram to {out_path}")
    print("Render it with mermaid-cli (mmdc) or paste into a Markdown viewer that supports Mermaid.")


if __name__ == "__main__":
    main()
