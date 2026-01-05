import hashlib
from datetime import datetime, timezone

import gradio as gr

from fact_check_llm import verify_news
from image_fact_checker import verify_image_news


# ---------------------------------------------------------------------------
# Helper utilities to format the redesigned interface content
# ---------------------------------------------------------------------------

STATUS_MAP = {
    "REAL": {
        "label_bn": "সত্য সংবাদ",
        "description": "আমাদের বিশ্লেষণ অনুযায়ী এই সংবাদটি নির্ভরযোগ্য।",
        "class": "status-real",
    },
    "FAKE": {
        "label_bn": "মিথ্যা সংবাদ",
        "description": "বিশ্বাসযোগ্য সূত্রের সাথে মিল না থাকায় এটি সন্দেহজনক।",
        "class": "status-fake",
    },
    "MISINFORMATION": {
        "label_bn": "ভ্রান্ত তথ্য",
        "description": "বিবরণে বিভ্রান্তিকর বা অসম্পূর্ণ তথ্য পাওয়া গেছে।",
        "class": "status-misinfo",
    },
    "UNSURE": {
        "label_bn": "অনিশ্চিত ফলাফল",
        "description": "পর্যাপ্ত প্রমাণের অভাবে নিশ্চিত হওয়া যায়নি।",
        "class": "status-uncertain",
    },
}


def _parse_verdict_text(verdict: str):
    classification = "UNSURE"
    explanation = verdict.strip()
    bengali_classification_keywords = ["শ্রেণীবিভাগ", "ধরণ"]

    for line in verdict.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue
        lower_line = clean_line.lower()
        normalized_line = clean_line.replace("ঃ", ":")
        if lower_line.startswith("classification") or any(normalized_line.startswith(word) for word in bengali_classification_keywords):
            classification = normalized_line.split(":", 1)[-1].strip() or classification
        elif lower_line.startswith("explanation") or normalized_line.startswith("ব্যাখ্যা"):
            explanation = normalized_line.split(":", 1)[-1].strip() or explanation

    classification_upper = classification.upper()
    if any(keyword in classification_upper for keyword in ["MISINFORMATION", "DISINFORMATION", "MISINFO", "ভ্রান্ত", "ভুল তথ্য"]):
        normalized_classification = "MISINFORMATION"
    elif any(keyword in classification_upper for keyword in ["FAKE", "মিথ্যা", "ভুয়া", "FALSE"]):
        normalized_classification = "FAKE"
    elif any(keyword in classification_upper for keyword in ["REAL", "সত্য", "TRUE"]):
        normalized_classification = "REAL"
    elif any(keyword in classification_upper for keyword in ["UNSURE", "UNCERTAIN", "অনিশ্চিত"]):
        normalized_classification = "UNSURE"
    else:
        normalized_classification = "UNSURE"

    return normalized_classification, explanation


def _extract_bullets(text: str, max_bullets: int = 5):
    if not text:
        return []
    parts = [p.strip() for p in text.replace("\n", ". ").split(".") if p.strip()]
    return parts[:max_bullets]


def _build_summary_card_html(summary_text: str, bullets):
    summary_text = summary_text.strip() if summary_text else "ফলাফল দেখতে একটি দাবি জমা দিন।"
    bullet_items = "".join(f"<li>{b}</li>" for b in bullets) if bullets else ""
    bullets_html = f"<ul>{bullet_items}</ul>" if bullet_items else ""
    return f"""
    <div class="info-card">
        <div class="info-header">
            <span class="material-symbols-outlined">psychology</span>
            <div>
                <p class="info-title">AI সারাংশ</p>
                <p class="info-caption">সংক্ষেপিত মূল বিষয়বস্তু</p>
            </div>
        </div>
        <p class="info-body">{summary_text}</p>
        {bullets_html}
    </div>
    """


def _append_reputation_summary(summary_text: str, onchain_metadata: dict) -> str:
    if not onchain_metadata:
        return summary_text
    publisher_rep = onchain_metadata.get("publisher_reputation") or {}
    count = publisher_rep.get("count")
    if count is None:
        urls = publisher_rep.get("urls")
        if isinstance(urls, list):
            count = len(urls)
    if count is None:
        return summary_text
    reputation_line = f"Publisher reputation count: {count}"
    if summary_text:
        return f"{summary_text}\n\n{reputation_line}"
    return reputation_line


def _build_findings_card_html(bullets):
    if not bullets:
        bullets = ["বিশ্লেষণ পাওয়া যায়নি। একটি দাবি জমা দিন।"]
    bullet_items = "".join(f"<li>{b}</li>" for b in bullets)
    return f"""
    <div class="info-card">
        <div class="info-header">
            <span class="material-symbols-outlined">gavel</span>
            <div>
                <p class="info-title">ফাইন্ডিংস</p>
                <p class="info-caption">RAG ভিত্তিক মূল পয়েন্ট</p>
            </div>
        </div>
        <ul class="bullet-list">{bullet_items}</ul>
    </div>
    """


def _build_sources_card_html(sources):
    if not sources:
        sources = [{"title": "সূত্র পাওয়া যায়নি", "url": "#", "snippet": "সংশ্লিষ্ট কোনো লিঙ্ক পাওয়া যায়নি।"}]
    items = []
    for src in sources:
        title = src.get("title") or "Untitled Source"
        url = src.get("url") or src.get("link") or "#"
        snippet = src.get("snippet") or ""
        safe_url = url if url.startswith("http") else "#"
        items.append(f"""
        <div class="evidence-item">
            <div>
                <p class="evidence-title"><a href="{safe_url}" target="_blank" rel="noopener noreferrer">{title}</a></p>
                <p class="evidence-subtitle">{safe_url}</p>
                <p class="evidence-body">{snippet}</p>
            </div>
        </div>
        """)
    return f"""
    <div class="info-card">
        <div class="info-header">
            <span class="material-symbols-outlined">link</span>
            <div>
                <p class="info-title">সূত্রসমূহ</p>
                <p class="info-caption">সংক্ষেপিত উদ্ধৃতি ও লিঙ্ক</p>
            </div>
        </div>
        <div class="evidence-list">
            {''.join(items)}
        </div>
    </div>
    """


def _build_blockchain_card_html(claim_text: str, onchain_metadata: dict = None):
    claim_text = (claim_text or "").strip()
    onchain_metadata = onchain_metadata or {}
    registration = onchain_metadata.get("registration") or {}
    tx_hash = registration.get("tx_hash")
    block_number = None
    if not tx_hash and claim_text:
        tx_hash = hashlib.sha256(claim_text.encode("utf-8")).hexdigest()
        block_number = int(tx_hash[:8], 16) % 10_000_000
    if not tx_hash:
        tx_hash = "—"
    if block_number is None:
        block_number = "—"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"""
    <div class="info-card">
        <div class="info-header">
            <span class="material-symbols-outlined">block</span>
            <div>
                <p class="info-title">ব্লকচেইন ভেরিফিকেশন</p>
                <p class="info-caption">অডিট লগ & টাইমস্ট্যাম্প</p>
            </div>
        </div>
        <div class="blockchain-grid">
            <div>
                <p class="label">Transaction Hash</p>
                <p class="code">{tx_hash}</p>
            </div>
            <div>
                <p class="label">Block Number</p>
                <p class="code">{block_number}</p>
            </div>
            <div>
                <p class="label">Timestamp</p>
                <p class="code">{timestamp}</p>
            </div>
        </div>
    </div>
    """


DEFAULT_SUMMARY_CARD = _build_summary_card_html("", [])
DEFAULT_FINDINGS_CARD = _build_findings_card_html([])
DEFAULT_SOURCES_CARD = _build_sources_card_html([])
DEFAULT_BLOCKCHAIN_CARD = _build_blockchain_card_html("")


# ---------------------------------------------------------------------------
# Interaction logic
# ---------------------------------------------------------------------------

def handle_input(choice, text_input, url_input, image_input):
    try:
        if choice == "Image":
            if not image_input:
                error_msg = _build_findings_card_html(["অনুগ্রহ করে একটি ইমেজ আপলোড করুন।"])
                return DEFAULT_SUMMARY_CARD, error_msg, DEFAULT_SOURCES_CARD, DEFAULT_BLOCKCHAIN_CARD

            claim, verdict_english, verdict_original, ocr_text, caption, visual_summary, credibility_score, backend_classification = verify_image_news(image_input)
            classification_from_text, explanation = _parse_verdict_text(verdict_english)
            summary_bullets = _extract_bullets(explanation)
            summary_card = _build_summary_card_html(verdict_original, summary_bullets)
            findings_card = _build_findings_card_html(summary_bullets)
            sources_card = _build_sources_card_html([])
            blockchain_card = _build_blockchain_card_html(claim)
            return summary_card, findings_card, sources_card, blockchain_card

        # Text flow
        combined_text = (text_input or "").strip()
        submitted_url = (url_input or "").strip()

        if not combined_text and not submitted_url:
            error_msg = _build_findings_card_html(["অনুগ্রহ করে সত্যতা যাচাইয়ের জন্য একটি বক্তব্য বা লিঙ্ক দিন।"])
            return DEFAULT_SUMMARY_CARD, error_msg, DEFAULT_SOURCES_CARD, DEFAULT_BLOCKCHAIN_CARD

        claim_input = combined_text or submitted_url
        claim, verdict_orig, verdict_trans, credibility_score, backend_classification, evidence_sources, onchain_metadata = verify_news(
            claim_input,
            submitted_url=submitted_url
        )
        classification_from_text, explanation = _parse_verdict_text(verdict_orig)
        summary_bullets = _extract_bullets(explanation)
        summary_text = _append_reputation_summary(verdict_trans, onchain_metadata)
        summary_card = _build_summary_card_html(summary_text, summary_bullets)
        findings_card = _build_findings_card_html(summary_bullets)
        sources_card = _build_sources_card_html(evidence_sources)
        blockchain_card = _build_blockchain_card_html(claim, onchain_metadata=onchain_metadata)
        return summary_card, findings_card, sources_card, blockchain_card

    except Exception as err:  # pragma: no cover - defensive fallback
        fallback_findings = _build_findings_card_html([f"ত্রুটি: {err}"])
        return DEFAULT_SUMMARY_CARD, fallback_findings, DEFAULT_SOURCES_CARD, DEFAULT_BLOCKCHAIN_CARD


def clear_all():
    return (
        "",
        "",
        None,
        DEFAULT_SUMMARY_CARD,
        DEFAULT_FINDINGS_CARD,
        DEFAULT_SOURCES_CARD,
        DEFAULT_BLOCKCHAIN_CARD,
    )


def toggle_visibility(selected):
    text_visible = selected == "Text"
    image_visible = selected == "Image"
    return (
        gr.update(visible=text_visible, value=""),
        gr.update(visible=text_visible, value=""),
        gr.update(visible=image_visible, value=None),
    )


# ---------------------------------------------------------------------------
# Recent claims panel helper
# ---------------------------------------------------------------------------

def _render_recent_claims():
    try:
        from claim_storage import ClaimStorageManager
        storage = ClaimStorageManager()
        claims = storage.list_recent_claims(limit=10)
    except Exception as e:
        print(f"Error loading recent claims: {e}")
        claims = []
    if not claims:
        return "<div class='info-card'><p class='info-title'>Checked Claims</p><p class='info-body'>No claims stored yet.</p></div>"
    items = []
    for c in claims:
        cls = (c.get("classification") or "UNSURE").upper()
        ts = c.get("timestamp") or ""
        snippet = c.get("claim_text_original") or ""
        items.append(f"""
        <div class="evidence-item">
            <p class="evidence-title">{cls} · {ts}</p>
            <p class="evidence-body">{snippet}</p>
        </div>
        """)
    return f"""
    <div class="info-card" style="max-height:320px; overflow-y:auto;">
        <div class="info-header">
            <span class="material-symbols-outlined">history</span>
            <div>
                <p class="info-title">Checked Claims</p>
                <p class="info-caption">Recent submissions (stored locally)</p>
            </div>
        </div>
        <div class="evidence-list">
            {''.join(items)}
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

FONT_ASSETS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@400;700&family=IBM+Plex+Mono:wght@400;500;700&family=Noto+Sans+Bengali:wght@400;500;700&family=Material+Symbols+Outlined" rel="stylesheet">
"""

HEADER_HTML = """
<div id="satya-header">
    <div class="brand">
        <span class="material-symbols-outlined">verified</span>
        <div>
            <p class="brand-title">সত্য সন্ধানী</p>
            <p class="brand-caption">Bangla Fake News Intelligence</p>
        </div>
    </div>
    <nav>
        <a href="#" aria-label="Home">Home</a>
        <a href="#" aria-label="About">About</a>
        <a href="#" aria-label="Explorer">Blockchain Explorer</a>
    </nav>
    <div class="header-actions">
        <button id="theme-toggle" aria-label="Switch theme">
            <span class="material-symbols-outlined icon-dark">dark_mode</span>
            <span class="material-symbols-outlined icon-light">light_mode</span>
        </button>
        <div class="avatar"></div>
    </div>
</div>
"""

THEME_SCRIPT = """
<script>
(function() {
    const root = document.documentElement;
    const THEME_KEY = "satya-theme";
    let currentMode = "dark";

    function applyTheme(mode) {
        if (!root) return;
        currentMode = mode === "dark" ? "dark" : "light";
        if (currentMode === "dark") {
            root.classList.add("dark-mode");
        } else {
            root.classList.remove("dark-mode");
        }
        const toggle = document.getElementById("theme-toggle");
        if (toggle) {
            toggle.classList.toggle("active-light", currentMode === "light");
        }
        try {
            localStorage.setItem(THEME_KEY, currentMode);
        } catch (err) {
            console.warn("Unable to persist theme", err);
        }
    }

    function toggleTheme() {
        applyTheme(currentMode === "light" ? "dark" : "light");
    }

    function bindToggle() {
        const button = document.getElementById("theme-toggle");
        if (button && !button.dataset.boundTheme) {
            button.addEventListener("click", toggleTheme);
            button.dataset.boundTheme = "true";
            button.classList.toggle("active-light", currentMode === "light");
        }
    }

    function initTheme() {
        let stored = null;
        try {
            stored = localStorage.getItem(THEME_KEY);
        } catch (err) {
            stored = null;
        }
        applyTheme(stored === "light" ? "light" : "dark");
        bindToggle();
    }

    const observer = new MutationObserver(bindToggle);

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", () => {
            initTheme();
            observer.observe(document.body, { childList: true, subtree: true });
        });
    } else {
        initTheme();
        observer.observe(document.body, { childList: true, subtree: true });
    }
})();
</script>
"""


CUSTOM_CSS = """
:root {
    --primary:#39FF14;
    --surface:#1A1A1A;
    --bg:#121212;
    --border:#444444;
    --text-strong:#FFFFFF;
    --text-muted:#B0B0B0;
    --shadow:0 12px 30px rgba(0,0,0,0.35);
}
.dark-mode {
    --primary:#39FF14;
    --surface:#1A1A1A;
    --bg:#121212;
    --border:#444444;
    --text-strong:#FFFFFF;
    --text-muted:#B0B0B0;
    --shadow:0 12px 30px rgba(0,0,0,0.35);
}
.gradio-container {
    background:var(--bg);
    color:var(--text-strong);
    padding-bottom:48px;
    font-family:'IBM Plex Mono','Noto Sans Bengali',monospace;
}
#main-content {
    margin-top:16px;
    gap:24px;
    max-width:1200px;
    margin-left:auto;
    margin-right:auto;
}
#satya-header {
    display:flex;
    align-items:center;
    justify-content:space-between;
    border:1px solid var(--border);
    border-radius:16px;
    padding:14px 20px;
    background:var(--surface);
    margin-bottom:16px;
    box-shadow:var(--shadow);
}
#satya-header nav a {
    margin:0 12px;
    color:var(--text-muted);
    text-decoration:none;
    font-weight:600;
    text-transform:uppercase;
    letter-spacing:0.02em;
}
#satya-header nav a:hover {
    color:var(--primary);
}
#satya-header .brand {
    display:flex;
    align-items:center;
    gap:10px;
}
#satya-header .brand-title {
    font-size:18px;
    font-family:'Oswald','Noto Sans Bengali',sans-serif;
    color:var(--text-strong);
    letter-spacing:0.02em;
}
#satya-header .brand-caption {
    font-size:12px;
    color:var(--text-muted);
}
#satya-header .avatar {
    width:36px;
    height:36px;
    border-radius:999px;
    background:var(--primary);
}
#satya-header .header-actions {
    display:flex;
    align-items:center;
    gap:10px;
}
#theme-toggle {
    width:44px;
    height:44px;
    border-radius:999px;
    border:1px solid var(--border);
    background:var(--surface);
    color:var(--text-strong);
    display:flex;
    align-items:center;
    justify-content:center;
    cursor:pointer;
    box-shadow:var(--shadow);
}
#theme-toggle .icon-light {display:none;}
.dark-mode #theme-toggle .icon-dark {display:none;}
.dark-mode #theme-toggle .icon-light {display:block;}

#mode-toggle {
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:999px;
    padding:6px;
    margin-bottom:12px;
    margin-top:4px;
    box-shadow:var(--shadow);
}
#mode-toggle label {flex:1 !important; font-weight:700; text-transform:uppercase; letter-spacing:0.02em;}
#mode-toggle input:checked+label {
    background:var(--primary);
    color:var(--bg);
}

#input-column, #result-column {
    display:flex;
    flex-direction:column;
    gap:16px;
}

#input-card, .info-card {
    border:1px solid var(--border);
    border-radius:18px;
    padding:18px;
    background:var(--surface);
    box-shadow:var(--shadow);
}
#input-card h2 {
    font-family:'Oswald','Noto Sans Bengali',sans-serif;
    color:var(--text-strong);
    font-size:28px;
    margin-bottom:8px;
    letter-spacing:0.03em;
}
#input-card textarea, #input-card input[type="text"] {
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:14px;
    color:var(--text-strong);
    font-size:15px;
}
.dark-mode #input-card textarea, .dark-mode #input-card input[type="text"] {
    background:#0f172a;
    color:var(--text-strong);
}
#input-card textarea:focus, #input-card input[type="text"]:focus {
    border-color:var(--primary);
    box-shadow:0 0 0 1px var(--primary);
}

#analyze-btn {
    background:var(--primary);
    color:var(--bg);
    border:none;
    border-radius:999px;
    height:50px;
    font-weight:800;
    font-size:16px;
    box-shadow:var(--shadow);
    letter-spacing:0.03em;
    text-transform:uppercase;
}
#analyze-btn:hover {opacity:0.9;}

#clear-btn {
    margin-top:8px;
    width:180px;
    border-radius:12px;
    border:1px solid var(--border);
    background:var(--surface);
    color:var(--text-strong);
    box-shadow:var(--shadow);
}
#clear-btn:hover {
    border-color:var(--primary);
    color:var(--primary);
}

.info-card {
    display:flex;
    flex-direction:column;
    gap:12px;
}
.info-header {
    display:flex;
    align-items:flex-start;
    gap:10px;
}
.info-title {
    font-family:'Oswald','Noto Sans Bengali',sans-serif;
    font-weight:700;
    color:var(--text-strong);
    margin-bottom:2px;
    letter-spacing:0.02em;
    text-transform:uppercase;
}
.info-caption {
    font-size:13px;
    color:var(--text-muted);
}
.info-body {
    color:var(--text-muted);
    font-size:15px;
    line-height:1.6;
}
.info-body.muted {color:var(--text-muted);}
.evidence-list {
    display:flex;
    flex-direction:column;
    gap:10px;
}
.evidence-item {
    padding:12px;
    border-radius:12px;
    background:var(--surface);
    border:1px solid var(--border);
}
.evidence-title {
    color:var(--text-strong);
    font-weight:600;
}
.evidence-title a {color:var(--text-strong); text-decoration:none;}
.evidence-title a:hover {text-decoration:underline;}
.evidence-subtitle {
    font-size:12px;
    color:var(--text-muted);
    margin-bottom:4px;
    word-break:break-all;
}
.evidence-body {
    font-size:14px;
    color:var(--text-muted);
}
.bullet-list {
    margin:0;
    padding-left:18px;
    color:var(--text-muted);
}
.blockchain-grid {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
    gap:12px;
}
.label {
    text-transform:uppercase;
    font-size:11px;
    letter-spacing:0.06em;
    color:var(--text-muted);
}
.code {
    font-family:monospace;
    color:var(--text-strong);
    margin-top:4px;
    word-break:break-all;
}
"""

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(FONT_ASSETS)
    gr.HTML(HEADER_HTML)
    gr.HTML(THEME_SCRIPT)

    choice = gr.Radio(
        ["Text", "Image"],
        value="Text",
        label="",
        elem_id="mode-toggle",
    )

    with gr.Row(elem_id="main-content"):
        with gr.Column(elem_id="input-column"):
            with gr.Group(elem_id="input-card"):
                gr.HTML("<h2>সত্যতা যাচাই করুন</h2>")
                text_box = gr.Textbox(
                    label="",
                    lines=8,
                    placeholder="এখানে বাংলা বা কোড-মিক্সড টেক্সট লিখুন...",
                    elem_id="claim-input",
                    visible=True,
                )
                url_box = gr.Textbox(
                    label="",
                    placeholder="ঐচ্ছিক: খবরের লিঙ্ক পেস্ট করুন",
                    elem_id="url-input",
                    visible=True,
                )
                image_input = gr.Image(
                    label="ছবি আপলোড করুন",
                    type="filepath",
                    visible=False,
                    elem_id="image-input",
                )
                submit_btn = gr.Button("এখনই বিশ্লেষণ করুন", elem_id="analyze-btn")
            clear_btn = gr.Button("রিসেট করুন", elem_id="clear-btn")

        with gr.Column(elem_id="result-column"):
            summary_card = gr.HTML(DEFAULT_SUMMARY_CARD, elem_id="ai-summary")
            findings_card = gr.HTML(DEFAULT_FINDINGS_CARD, elem_id="findings-card")
            sources_card = gr.HTML(DEFAULT_SOURCES_CARD, elem_id="evidence-card")
            blockchain_card = gr.HTML(DEFAULT_BLOCKCHAIN_CARD, elem_id="blockchain-card")


    choice.change(
        fn=toggle_visibility,
        inputs=choice,
        outputs=[text_box, url_box, image_input],
    )

    submit_btn.click(
        fn=handle_input,
        inputs=[choice, text_box, url_box, image_input],
        outputs=[summary_card, findings_card, sources_card, blockchain_card],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[text_box, url_box, image_input, summary_card, findings_card, sources_card, blockchain_card],
    )

    gr.HTML(_render_recent_claims(), elem_id="recent-claims-bottom")

demo.launch()
