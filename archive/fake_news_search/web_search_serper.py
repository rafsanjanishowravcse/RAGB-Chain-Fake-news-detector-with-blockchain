import os
import requests
from typing import List, Dict

SERPER_API_KEY = "2931bbdd6b21da755b6bf2794ba2c33134e8a2bb"
if not SERPER_API_KEY:
    raise RuntimeError(
        "Missing SERPER_API_KEY environment variable. "
        "Obtain one from https://serper.dev/ and set it before running."
    )

# Serper.dev search endpoint
_SERPER_SEARCH_URL = "https://google.serper.dev/search"

def search_serper(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Query Serper.dev and return up to `num_results` organic search hits.
    Each hit is a dict: { "title": str, "link": str, "snippet": str }.
    Raises HTTPError on non-2xx responses.
    """
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "num": num_results,
    }

    resp = requests.post(_SERPER_SEARCH_URL, headers=headers, json=payload, timeout=5)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("organic", [])[:num_results]:
        title = item.get("title", "").strip()
        link = item.get("link", "").strip()
        snippet = item.get("snippet", "").strip()
        results.append({"title": title, "link": link, "snippet": snippet})
    return results

# Quick test when running this file directly
if __name__ == "__main__":
    print("=== Serper.dev Search Test ===")
    hits = search_serper("site: news18 Vaccines cause autism", num_results=3)
    for i, h in enumerate(hits, 1):
        print(f"{i}. {h['title']}\n   URL: {h['link']}\n   Snippet: {h['snippet']}\n")
