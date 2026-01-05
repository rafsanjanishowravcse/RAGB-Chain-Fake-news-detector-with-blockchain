import requests
from newspaper import Article
from bs4 import BeautifulSoup
from typing import List, Dict

from web_search_serper import search_serper

def scrape_full_text(url: str, max_chars: int = 2000) -> str:
    """
    Try Newspaper3k first; on failure, fall back to BeautifulSoup.
    Returns up to `max_chars` of raw text.
    """
    try:
        art = Article(url)
        art.download()
        art.parse()
        text = art.text
        if text and len(text) > 50:
            return text[:max_chars]
    except Exception:
        pass

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paras = soup.find_all("p")
        joined = " ".join(p.get_text() for p in paras)
        return joined[:max_chars]
    except Exception:
        return ""

def retrieve_evidence_for_claim(claim: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    For a given `claim` string, run Serper search and scrape each result.
    
    Returns a list of dicts:
      {
        "url": str,
        "title": str,
        "snippet": str,
        "raw_text": str
      }
    """
    hits = search_serper(claim, num_results)
    evidence = []

    for item in hits:
        url    = item["link"]
        title  = item["title"]
        snippet= item["snippet"]
        raw    = scrape_full_text(url)

        evidence.append({
            "url": url,
            "title": title,
            "snippet": snippet,
            "raw_text": raw
        })

    return evidence

# Quick test when running this file directly
if __name__ == "__main__":
    print("=== Retrieval Pipeline Test ===")
    claim = "Vaccines cause autism"
    evs = retrieve_evidence_for_claim(claim, num_results=3)
    for i, e in enumerate(evs, 1):
        print(f"{i}. {e['title']} ({e['url']})")
        print(f"   Snippet: {e['snippet'][:100]}...")
        print(f"   Raw Text Start: {e['raw_text'][:150]}...\n")
